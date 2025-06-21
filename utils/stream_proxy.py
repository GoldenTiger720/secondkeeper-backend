# utils/stream_proxy.py

import os
import threading
import subprocess
import logging
import time
import signal
import tempfile
from pathlib import Path
from django.conf import settings

logger = logging.getLogger(__name__)

class StreamProxyManager:
    """
    Singleton class to manage multiple stream proxies
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(StreamProxyManager, cls).__new__(cls)
                    cls._instance.active_streams = {}
        return cls._instance
    
    def get_or_create_stream(self, camera):
        """
        Get existing stream or create new one for camera
        """
        camera_id = str(camera.id)
        
        if camera_id not in self.active_streams:
            self.active_streams[camera_id] = StreamProxy(camera)
        
        return self.active_streams[camera_id]
    
    def stop_all_except(self, keep_camera_id):
        """
        Stop all streams except the specified camera ID
        """
        keep_camera_id = str(keep_camera_id)
        
        for camera_id, stream_proxy in list(self.active_streams.items()):
            if camera_id != keep_camera_id:
                logger.info(f"Stopping stream for camera {camera_id}")
                stream_proxy.force_cleanup()
                del self.active_streams[camera_id]
    
    def stop_all(self):
        """
        Stop all active streams
        """
        for camera_id, stream_proxy in list(self.active_streams.items()):
            logger.info(f"Stopping stream for camera {camera_id}")
            stream_proxy.force_cleanup()
        
        self.active_streams.clear()
    
    def cleanup_orphaned_processes(self):
        """
        Clean up any orphaned FFmpeg processes
        """
        try:
            processes = subprocess.check_output(['ps', 'aux']).decode()
            for line in processes.split('\n'):
                if 'ffmpeg' in line and 'streams' in line:
                    pid = line.split()[1]
                    try:
                        subprocess.run(['kill', '-9', pid])
                        logger.info(f"Killed orphaned FFmpeg process {pid}")
                    except Exception as e:
                        logger.error(f"Error killing orphaned process: {e}")
        except Exception as e:
            logger.error(f"Error cleaning up orphaned processes: {e}")

class StreamProxy:
    """
    Class for proxying video streams to web-friendly formats (HLS/DASH)
    using FFmpeg.
    """
    
    def __init__(self, camera):
        """
        Initialize the stream proxy
        
        Args:
            camera: Camera model instance
        """
        self.camera = camera
        self.process = None
        self.is_running = False
        self.output_dir = os.path.join(settings.MEDIA_ROOT, 'streams', str(camera.id))
        self.stop_event = threading.Event()
        self.hls_playlist_path = os.path.join(self.output_dir, 'index.m3u8')
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def cleanup_stream_files(self):
        """
        Clean up all stream files for this camera
        """
        try:
            if os.path.exists(self.output_dir):
                for filename in os.listdir(self.output_dir):
                    file_path = os.path.join(self.output_dir, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                            logger.debug(f"Deleted {file_path}")
                    except Exception as e:
                        logger.error(f"Failed to delete {file_path}: {e}")
                
                logger.info(f"Cleaned up stream files for camera {self.camera.id}")
        except Exception as e:
            logger.error(f"Error cleaning up stream files: {e}")
    
    def kill_existing_processes(self):
        """
        Kill any existing FFmpeg processes for this camera
        """
        try:
            processes = subprocess.check_output(['ps', 'aux']).decode()
            killed_count = 0
            
            for line in processes.split('\n'):
                if 'ffmpeg' in line and str(self.camera.id) in line:
                    pid = line.split()[1]
                    try:
                        subprocess.run(['kill', '-9', pid], check=True)
                        logger.info(f"Killed existing FFmpeg process {pid} for camera {self.camera.id}")
                        killed_count += 1
                    except Exception as e:
                        logger.error(f"Error killing process {pid}: {e}")
            
            if killed_count > 0:
                # Wait a moment for processes to fully terminate
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"Error checking for existing processes: {e}")
    
    def check_hls_ready(self, timeout=30):
        """
        Check if HLS playlist and segments are ready
        
        Args:
            timeout: Maximum time to wait for HLS files (seconds)
            
        Returns:
            dict: Status of HLS readiness
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # Check if index.m3u8 exists
                if os.path.exists(self.hls_playlist_path):
                    # Check if the playlist has content
                    with open(self.hls_playlist_path, 'r') as f:
                        content = f.read()
                        
                    # Check if playlist contains at least one segment
                    if '.ts' in content and '#EXTINF' in content:
                        # Verify that at least one segment file actually exists
                        lines = content.split('\n')
                        segment_found = False
                        
                        for line in lines:
                            if line.strip().endswith('.ts'):
                                segment_path = os.path.join(self.output_dir, line.strip())
                                if os.path.exists(segment_path) and os.path.getsize(segment_path) > 0:
                                    segment_found = True
                                    break
                        
                        if segment_found:
                            logger.info(f"HLS stream ready for camera {self.camera.id}")
                            return {
                                'ready': True,
                                'playlist_path': self.hls_playlist_path,
                                'url': f"/media/streams/{self.camera.id}/index.m3u8"
                            }
                
                # Wait a bit before checking again
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error checking HLS readiness: {e}")
                time.sleep(0.5)
        
        return {
            'ready': False,
            'error': f"HLS playlist not ready after {timeout} seconds"
        }

    def start(self, target_fps=15, force_restart=True):
        """
        Start the stream proxy and wait for HLS files to be ready
        
        Args:
            target_fps: Target frame rate for output (default: 15)
            force_restart: Force restart even if already running (default: True)
        
        Returns:
            dict: Success status, message, and stream URL
        """
        try:
            logger.info(f"Starting stream for camera {self.camera.id}")
            
            # Always kill existing processes and clean files when force_restart is True
            if force_restart:
                logger.info(f"Force restart requested for camera {self.camera.id}")
                
                # Kill any existing FFmpeg processes for this camera
                self.kill_existing_processes()
                
                # Stop current process if running
                if self.process:
                    try:
                        self.process.terminate()
                        self.process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                    except Exception as e:
                        logger.error(f"Error stopping current process: {e}")
                
                # Clean up all existing files
                self.cleanup_stream_files()
                
                # Reset state
                self.is_running = False
                self.process = None
                self.stop_event.clear()
            
            # Prepare input URL based on camera type
            input_url = self._get_input_url()
            logger.info(f"Input URL: {input_url}")
            logger.info(f"Target FPS: {target_fps}")

            # Calculate GOP size (2 seconds worth of frames)
            gop_size = target_fps * 2

            # Prepare FFmpeg command for HLS streaming
            cmd = [
                'ffmpeg',
                '-y',                            # Overwrite output files
                '-i', input_url,                 # Input stream
                '-r', str(target_fps),           # INPUT frame rate
                '-c:v', 'libx264',               # Video codec
                '-preset', 'veryfast',           # Encoding preset
                '-tune', 'zerolatency',          # Tuning for low latency
                '-sc_threshold', '0',            # Disable scene change detection
                '-g', str(gop_size),             # GOP size (2 seconds)
                '-r', str(target_fps),           # OUTPUT frame rate
                '-hls_time', '2',                # Segment length in seconds
                '-hls_list_size', '5',           # Number of segments in playlist
                '-hls_flags', 'delete_segments', # Delete old segments
                '-hls_segment_type', 'mpegts',   # Segment type
                '-hls_segment_filename', f"{self.output_dir}/segment_%03d.ts", # Segment filename pattern
                '-f', 'hls',                     # Output format (HLS)
                self.hls_playlist_path           # Output playlist
            ]
            
            logger.info(f"FFmpeg command: {' '.join(cmd)}")
            
            # Start FFmpeg process
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Wait a moment to see if process starts successfully
            time.sleep(2)
            
            # Check if process is still running
            if self.process.poll() is None:
                self.is_running = True
                
                # Start background thread to monitor process
                self._start_monitor_thread()
                
                logger.info(f"FFmpeg process started for camera {self.camera.id} (PID: {self.process.pid})")
                
                # Wait for HLS files to be ready
                hls_status = self.check_hls_ready(timeout=30)
                
                if hls_status['ready']:
                    logger.info(f"Stream successfully started for camera {self.camera.id}")
                    return {
                        'success': True,
                        'message': 'Stream proxy started successfully',
                        'stream_url': hls_status['url'],
                        'hls_ready': True,
                        'playlist_path': hls_status['playlist_path']
                    }
                else:
                    # HLS files not ready, stop the process
                    logger.error(f"HLS files not ready for camera {self.camera.id}")
                    self.force_cleanup()
                    return {
                        'success': False,
                        'message': f"HLS stream not ready: {hls_status.get('error', 'Unknown error')}",
                        'hls_ready': False
                    }
            else:
                # Process has already exited, get error details
                stdout, stderr = self.process.communicate()
                logger.error(f"FFmpeg failed to start for camera {self.camera.id}. STDOUT: {stdout}, STDERR: {stderr}")
                return {
                    'success': False,
                    'message': f"Stream proxy start failed: {stderr}",
                    'hls_ready': False
                }
            
        except Exception as e:
            logger.error(f"Error starting stream proxy for camera {self.camera.id}: {str(e)}")
            self.force_cleanup()
            return {
                'success': False,
                'message': f"Error starting stream proxy: {str(e)}",
                'hls_ready': False
            }
    
    def stop(self):
        """
        Stop the stream proxy gracefully
        """
        try:
            logger.info(f"Stopping stream for camera {self.camera.id}")
            
            # Set stop event to signal monitoring thread
            self.stop_event.set()
            
            if self.process:
                try:
                    self.process.terminate()
                    self.process.wait(timeout=5)
                    logger.info(f"FFmpeg process terminated for camera {self.camera.id}")
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    logger.info(f"FFmpeg process killed for camera {self.camera.id}")
                except Exception as e:
                    logger.error(f"Error stopping FFmpeg process: {e}")
            
            self.is_running = False
            self.process = None
            
            # Reset stop event for next start
            self.stop_event.clear()
            
        except Exception as e:
            logger.error(f"Error stopping stream proxy: {e}")
    
    def force_cleanup(self):
        """
        Force cleanup of all stream files and processes
        """
        logger.info(f"Force cleanup for camera {self.camera.id}")
        
        # Kill existing processes
        self.kill_existing_processes()
        
        # Stop current process
        self.stop()
        
        # Clean up files
        self.cleanup_stream_files()
    
    def _get_input_url(self):
        """
        Get the input URL for FFmpeg based on camera type
        
        Returns:
            str: Input URL
        """
        # For RTSP cameras
        if self.camera.stream_url:
            stream_url = self.camera.stream_url
            return stream_url
        else:
            raise ValueError(f"Unsupported camera URL: {self.camera.stream_url}")
    
    def _start_monitor_thread(self):
        """
        Start a background thread to monitor the FFmpeg process
        """
        def monitor_process():
            while not self.stop_event.is_set() and self.is_running:
                # Check if process is still running
                if self.process and self.process.poll() is not None:
                    # Process has exited, get the error output
                    try:
                        stdout, stderr = self.process.communicate(timeout=1)
                        if stderr:
                            logger.error(f"Stream proxy exited with error for camera {self.camera.id}: {stderr}")
                        if stdout:
                            logger.debug(f"Stream proxy output for camera {self.camera.id}: {stdout}")
                    except subprocess.TimeoutExpired:
                        pass
                    
                    self.is_running = False
                    logger.warning(f"Stream process ended for camera {self.camera.id}")
                    break
                
                # Sleep for a bit
                time.sleep(5)
        
        # Start the monitor thread
        monitor_thread = threading.Thread(target=monitor_process, name=f"StreamMonitor-{self.camera.id}")
        monitor_thread.daemon = True
        monitor_thread.start()

# Convenience functions for easy usage
def start_camera_stream(camera, target_fps=15):
    """
    Start stream for a camera, stopping all other streams
    
    Args:
        camera: Camera model instance
        target_fps: Target frame rate
        
    Returns:
        dict: Stream start result
    """
    manager = StreamProxyManager()
    
    # Stop all other streams
    manager.stop_all_except(camera.id)
    
    # Clean up any orphaned processes
    manager.cleanup_orphaned_processes()
    
    # Get or create stream for this camera
    stream_proxy = manager.get_or_create_stream(camera)
    
    # Start the stream with force restart
    return stream_proxy.start(target_fps=target_fps, force_restart=True)

def stop_all_streams():
    """
    Stop all active streams
    """
    manager = StreamProxyManager()
    manager.stop_all()
    manager.cleanup_orphaned_processes()