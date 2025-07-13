# utils/rtsp_stream.py

import subprocess
import logging
import threading
from django.http import StreamingHttpResponse

logger = logging.getLogger(__name__)

class RTSPDirectStream:
    """
    Direct RTSP to MP4 streaming for real-time camera feeds.
    Based on Flask implementation but adapted for Django.
    """
    
    def __init__(self, camera):
        """
        Initialize the RTSP direct stream
        
        Args:
            camera: Camera model instance
        """
        self.camera = camera
        self.process = None
        self.is_running = False
        
    def generate_stream(self):
        """
        Generator function that yields MP4 chunks from FFmpeg process
        
        Yields:
            bytes: MP4 video chunks
        """
        try:
            # Build FFmpeg command for direct MP4 streaming
            ffmpeg_cmd = [
                "ffmpeg",
                "-rtsp_transport", "tcp",
                "-i", self.camera.stream_url,
                "-vcodec", "copy",
                "-f", "mp4",
                "-movflags", "frag_keyframe+empty_moov",
                "-reset_timestamps", "1",
                "-vsync", "1",
                "-flags", "global_header",
                "-bsf:v", "dump_extra",
                "-y", "-"
            ]
            
            logger.info(f"Starting direct RTSP stream for camera {self.camera.id}: {self.camera.name}")
            logger.info(f"FFmpeg command: {' '.join(ffmpeg_cmd)}")
            
            # Start FFmpeg process
            self.process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=-1
            )
            
            self.is_running = True
            
            # Start thread to monitor stderr
            stderr_thread = threading.Thread(
                target=self._monitor_stderr,
                daemon=True
            )
            stderr_thread.start()
            
            try:
                while self.is_running:
                    # Read chunks from FFmpeg stdout
                    chunk = self.process.stdout.read(1024)
                    if not chunk:
                        logger.warning(f"No more data from FFmpeg for camera {self.camera.id}")
                        break
                    yield chunk
                    
            except GeneratorExit:
                logger.info(f"Client closed connection for camera {self.camera.id}")
                self.stop()
            except Exception as e:
                logger.error(f"Error in stream generation: {e}")
                self.stop()
            finally:
                self.stop()
                
        except Exception as e:
            logger.error(f"Error starting RTSP stream: {e}")
            self.stop()
            
    def _monitor_stderr(self):
        """
        Monitor FFmpeg stderr output for debugging
        """
        try:
            for line in iter(self.process.stderr.readline, b''):
                if not self.is_running:
                    break
                if line:
                    logger.debug(f"FFmpeg stderr: {line.decode('utf-8', errors='ignore').strip()}")
        except Exception as e:
            logger.error(f"Error monitoring stderr: {e}")
            
    def stop(self):
        """
        Stop the stream and clean up resources
        """
        try:
            self.is_running = False
            
            if self.process:
                logger.info(f"Stopping FFmpeg process for camera {self.camera.id}")
                
                # Try graceful termination first
                try:
                    self.process.terminate()
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    # Force kill if termination fails
                    logger.warning(f"Force killing FFmpeg process for camera {self.camera.id}")
                    self.process.kill()
                    self.process.wait()
                except Exception as e:
                    logger.error(f"Error stopping FFmpeg process: {e}")
                    
                self.process = None
                logger.info(f"FFmpeg process stopped for camera {self.camera.id}")
                
        except Exception as e:
            logger.error(f"Error in stop method: {e}")


def create_rtsp_streaming_response(camera):
    """
    Create a Django StreamingHttpResponse for RTSP camera
    
    Args:
        camera: Camera model instance
        
    Returns:
        StreamingHttpResponse: Streaming response with MP4 video
    """
    stream = RTSPDirectStream(camera)
    
    response = StreamingHttpResponse(
        stream.generate_stream(),
        content_type='video/mp4'
    )
    
    # Set headers (excluding hop-by-hop headers that Django doesn't allow)
    response['Accept-Ranges'] = 'bytes'
    response['Access-Control-Allow-Origin'] = '*'
    response['Access-Control-Request-Method'] = '*'
    response['Access-Control-Allow-Methods'] = 'OPTIONS, GET'
    response['Access-Control-Allow-Headers'] = '*'
    response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response['Pragma'] = 'no-cache'
    response['Expires'] = '0'
    
    return response