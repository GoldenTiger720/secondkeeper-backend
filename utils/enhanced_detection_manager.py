# utils/enhanced_detection_manager.py - COMPLETE FIX with proper alert storage and video recording

import cv2
import threading
import time
import os
import logging
import glob
import torch
from datetime import datetime
from django.conf import settings
from django.utils import timezone
from cameras.models import Camera
from detectors.fire_detector_class import FireSmokeDetector
from detectors.choking_detector_class import ChokingDetector
from detectors.fall_detector_class import FallDetector
from detectors.violence_detector_class import ViolenceDetector

logger = logging.getLogger('security_ai')

class EnhancedDetectionManager:
    """
    Enhanced detection manager that prioritizes video files over camera streams
    """
    
    def __init__(self):
        self.active_cameras = {}  # camera_id -> CameraProcessor
        self.active_video_processors = {}  # video_file -> VideoFileProcessor
        self.is_running = False
        self.main_thread = None
        self.stop_event = threading.Event()
        
        # Test video directory
        self.test_video_dir = os.path.join(settings.MEDIA_ROOT, 'testvideo')
        os.makedirs(self.test_video_dir, exist_ok=True)
        # GPU optimization
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Detection settings
        self.frame_skip = 5  # Process every 5th frame for performance
        self.enabled_detectors = ['fire', 'fall', 'choking', 'violence']
        self.video_file = None  # Current video file being processed
        
    def start(self):
        """Start the enhanced detection manager"""
        if self.is_running:
            print("Enhanced detection manager is already running")
            return
        self.is_running = True
        self.stop_event.clear()
        
        # Start main processing thread
        self.main_thread = threading.Thread(target=self._main_loop, daemon=True)
        self.main_thread.start()
        
    def stop(self):
        """Stop the enhanced detection manager"""
        if not self.is_running:
            return
            
        logger.info("Stopping Enhanced Detection Manager")
        self.is_running = False
        self.stop_event.set()
        
        # Stop all processors
        for processor in self.active_cameras.values():
            processor.stop()
            
        for processor in self.active_video_processors.values():
            processor.stop()
            
        # Wait for main thread to finish
        if self.main_thread and self.main_thread.is_alive():
            self.main_thread.join(timeout=10)
            
        self.active_cameras.clear()
        self.active_video_processors.clear()
        
    def display_frame(self, detector_config, stop_event=None):
        detector_class = None
        cap = None
        try:
            detector_type = detector_config['detector_type']
            rtsp_url = detector_config.get('rtsp_url', '')
            video_file = detector_config.get('video_file', 'video.mp4')
            use_rtsp = detector_config.get('use_rtsp', False)
            camera_id = detector_config.get('camera_id', None)
            
            # Create detector instance based on type with proper parameters
            if detector_type == 'fire':
                detector_class = FireSmokeDetector(camera_id=camera_id, rtsp_url=rtsp_url, video_file=video_file, use_rtsp=use_rtsp)
            elif detector_type == 'fall':
                detector_class = FallDetector(camera_id=camera_id, rtsp_url=rtsp_url, video_file=video_file, use_rtsp=use_rtsp)
            elif detector_type == 'choking':
                detector_class = ChokingDetector(camera_id=camera_id, rtsp_url=rtsp_url, video_file=video_file, use_rtsp=use_rtsp)
            elif detector_type == 'violence':
                detector_class = ViolenceDetector(camera_id=camera_id, rtsp_url=rtsp_url, video_file=video_file, use_rtsp=use_rtsp)
            else:
                print(f"[ERROR] Unknown detector type: {detector_type}")
                return
            
            cap = cv2.VideoCapture(detector_class.source)
            if not cap.isOpened():
                print(f"[ERROR] Cannot open video: {detector_class.source}")
                return

            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            frame_count = 0
            while not (stop_event and stop_event.is_set()):
                ret, frame = cap.read()
                if not ret:
                    # If it's a video file, loop it
                    if not detector_config['use_rtsp']:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        print(f"[INFO] {detector_type.capitalize()} Detector: End of stream reached")
                        break

                frame = detector_class.process_frame(frame, fps)
                frame_count += 1
                if frame_count % 300 == 0:  # Log every 300 frames (10 seconds at 30fps)
                    print(f"[INFO] {detector_type.capitalize()} Detector processed {frame_count} frames")
                time.sleep(0.01)
                    
        except KeyboardInterrupt:
            print(f"[INFO] {detector_type.capitalize()} Detector: Stopping detection...")
        except Exception as e:
            print(f"[ERROR] Error in {detector_type.capitalize()} Detector: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if cap is not None:
                cap.release()
            cv2.destroyAllWindows()
            # Clean up detector resources
            if detector_class is not None:
                detector_class.cleanup()
            print(f"[INFO] {detector_config.get('detector_type', 'Unknown')} Detector cleanup completed")

    def run_detectors(self):
        """Run detectors for current video file (backward compatibility)"""
        enabled_detectors = ['violence', 'fire', 'fall', 'choking']
        use_rtsp = False  # Video files do not use RTSP
        rtsp_url = ""  
        detector_configs = []
        
        for detector_type in enabled_detectors:
            detector_config = {
                'detector_type': detector_type,
                'rtsp_url': rtsp_url,
                'video_file': self.video_file,
                'use_rtsp': use_rtsp,
                'camera_id': None
            }
            detector_configs.append(detector_config)
            print(f"[INFO] Added {detector_type.capitalize()} Detector to configuration")

        self._run_detector_threads(detector_configs)
        
    def _run_detector_threads(self, detector_configs):
        """Run detector threads with given configurations"""
        if not detector_configs:
            print("[ERROR] No detectors configured!")
            return

        # Create a thread for each detector configuration
        threads = []
        stop_events = []
        
        for detector_config in detector_configs:
            stop_event = threading.Event()
            stop_events.append(stop_event)
            
            thread = threading.Thread(
                target=self.display_frame, 
                args=(detector_config, stop_event),
                name=f"{detector_config['detector_type'].capitalize()}DetectorThread",
                daemon=True
            )
            threads.append(thread)
            thread.start()
            print(f"[INFO] Started {detector_config['detector_type'].capitalize()} Detector thread")

        try:
            print(f"[INFO] Running {len(threads)} detector threads. Press Ctrl+C to stop.")
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
        except KeyboardInterrupt:
            print("[INFO] Stopping all detector threads...")
            for stop_event in stop_events:
                stop_event.set()
            
            # Wait for threads to terminate gracefully
            for thread in threads:
                thread.join(timeout=5)
                if thread.is_alive():
                    print(f"[WARNING] Thread {thread.name} still running")
        
        print("[INFO] All detector threads have been stopped.")
        
    def _main_loop(self):
        """Main processing loop with video file priority"""
        while self.is_running and not self.stop_event.is_set():
            try:
                # Step 1: Check for video files in test directory
                video_files = self._get_test_video_files()
                
                if video_files:
                    logger.info(f"Found {len(video_files)} test video files, prioritizing video processing")
                    self._process_video_files(video_files)
                    # Stop camera processors when processing video files
                    self._stop_all_camera_processors()
                else:
                    logger.info("No test video files found, switching to camera detection")
                    # Stop video processors
                    self._stop_all_video_processors()
                    # Process camera streams
                    self._process_camera_streams()
                
            except Exception as e:
                logger.error(f"Error in main enhanced detection loop: {str(e)}")
            
            # Sleep for a bit before next iteration
            time.sleep(5)
    def _get_test_video_files(self):
        """Get list of video files in test directory"""
        try:
            video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv', '*.webm']
            video_files = []
            
            for extension in video_extensions:
                pattern = os.path.join(self.test_video_dir, extension)
                video_files.extend(glob.glob(pattern))
            
            # Sort by modification time (newest first)
            video_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
            return video_files
            
        except Exception as e:
            logger.error(f"Error getting test video files: {str(e)}")
            return []
            
    def _process_video_files(self, video_files):
        """Process video files for detection"""
        current_processors = set(self.active_video_processors.keys())
        new_video_files = set(video_files)
        
        # Stop processors for videos that no longer exist
        videos_to_remove = current_processors - new_video_files
        for video_file in videos_to_remove:
            self._stop_video_processor(video_file)
        
        # Start processors for new video files
        videos_to_add = new_video_files - current_processors
        for video_file in videos_to_add:
            if os.path.exists(video_file):
                self._start_video_processor(video_file)
                
    def _process_camera_streams(self):
        """Process camera streams when no video files are present"""
        # Get online cameras from database
        online_cameras = Camera.objects.filter(
            status='online',
            detection_enabled=True
        ).select_related('user')
        
        current_camera_ids = set(self.active_cameras.keys())
        new_camera_ids = set(str(cam.id) for cam in online_cameras)
        
        # Stop processors for cameras that are no longer online
        cameras_to_remove = current_camera_ids - new_camera_ids
        for camera_id in cameras_to_remove:
            self._stop_camera_processor(camera_id)
        
        # Start processors for new cameras
        cameras_to_add = new_camera_ids - current_camera_ids
        for camera in online_cameras:
            if str(camera.id) in cameras_to_add:
                self._start_camera_processor(camera)
        
        # Update existing processors with new camera data
        for camera in online_cameras:
            camera_id = str(camera.id)
            if camera_id in self.active_cameras:
                self.active_cameras[camera_id].update_camera_data(camera)
                
    def _start_video_processor(self, video_file):
        """Start a processor for a video file"""
        try:
            enabled_detectors = ['violence', 'fire', 'fall', 'choking']
            use_rtsp = False  # Video files do not use RTSP
            rtsp_url = ""
            self.video_file = video_file  # Set the current video file
            
            # Create detector configurations for video processing
            detector_configs = []
            for detector_type in enabled_detectors:
                detector_config = {
                    'detector_type': detector_type,
                    'rtsp_url': rtsp_url,
                    'video_file': video_file,
                    'use_rtsp': use_rtsp,
                    'camera_id': None
                }
                detector_configs.append(detector_config)
                print(f"[INFO] Added {detector_type.capitalize()} Detector for video file: {video_file}")

            # Start detector threads for video processing
            self._run_detector_threads(detector_configs)
            
        except Exception as e:
            logger.error(f"Error starting video processor for {video_file}: {str(e)}")
            
    def _stop_video_processor(self, video_file):
        """Stop a processor for a video file"""
        try:
            if video_file in self.active_video_processors:
                logger.info(f"Stopping video processor for file: {video_file}")
                self.active_video_processors[video_file].stop()
                del self.active_video_processors[video_file]
                
        except Exception as e:
            logger.error(f"Error stopping video processor for {video_file}: {str(e)}")
            
    def _start_camera_processor(self, camera):
        """Start a processor for a single camera"""
        try:
            camera_id = str(camera.id)
            logger.info(f"Starting processor for camera {camera_id} - {camera.name}")
            
            enabled_detectors = ['violence', 'fire', 'fall', 'choking']
            use_rtsp = True  # Cameras use RTSP streams
            rtsp_url = camera.stream_url
            video_file = ""
            
            # Create detector configurations for camera processing
            detector_configs = []
            for detector_type in enabled_detectors:
                detector_config = {
                    'detector_type': detector_type,
                    'rtsp_url': rtsp_url,
                    'video_file': video_file,
                    'use_rtsp': use_rtsp,
                    'camera_id': camera.id
                }
                detector_configs.append(detector_config)
                print(f"[INFO] Added {detector_type.capitalize()} Detector for camera: {camera.name}")

            # Start detector threads for camera processing
            self._run_detector_threads(detector_configs)
            
        except Exception as e:
            logger.error(f"Error starting processor for camera {camera.id}: {str(e)}")
            
    def _stop_camera_processor(self, camera_id):
        """Stop a processor for a single camera"""
        try:
            if camera_id in self.active_cameras:
                logger.info(f"Stopping processor for camera {camera_id}")
                self.active_cameras[camera_id].stop()
                del self.active_cameras[camera_id]
                
        except Exception as e:
            logger.error(f"Error stopping processor for camera {camera_id}: {str(e)}")
            
    def _stop_all_camera_processors(self):
        """Stop all camera processors"""
        for camera_id in list(self.active_cameras.keys()):
            self._stop_camera_processor(camera_id)
            
    def _stop_all_video_processors(self):
        """Stop all video processors"""
        for video_file in list(self.active_video_processors.keys()):
            self._stop_video_processor(video_file)


class EnhancedCameraProcessor:
    """
    Enhanced processor for camera streams
    """
    
    def __init__(self, camera, app, device, manager):
        self.camera = camera
        self.app = app
        self.device = device
        self.manager = manager
        
        self.is_running = False
        self.thread = None
        self.cap = None
        self.frame_count = 0
        
    def start(self):
        """Start processing this camera"""
        if self.is_running:
            return
            
        self.is_running = True
        # self.thread = threading.Thread(target=self._process_loop, daemon=True)
        # self.thread.start()
        self.app.run_detection_threads(self.camera.stream_url)
        
    def stop(self):
        """Stop processing this camera"""
        self.is_running = False
        if self.cap:
            self.cap.release()
            
    def update_camera_data(self, camera):
        """Update camera data"""
        self.camera = camera
        
    def _process_loop(self):
        """Main processing loop for this camera"""
        try:
            # Open camera stream
            self.cap = cv2.VideoCapture(self.camera.stream_url)
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.camera.id}: {self.camera.stream_url}")
                self._update_camera_status('offline')
                return
                
            self._update_camera_status('online')
            logger.info(f"Started processing camera {self.camera.id} - {self.camera.name}")
            
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning(f"Failed to read frame from camera {self.camera.id}")
                    time.sleep(1)
                    continue
                    
                self.frame_count += 1
                fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 20
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # Skip frames for performance
                if self.frame_count % self.manager.frame_skip != 0:
                    continue

                # Process frame with all enabled detectors
                self._process_frame(frame, fps, width, height)

        except Exception as e:
            logger.error(f"Error in camera {self.camera.id} processing loop: {str(e)}")
            self._update_camera_status('error')
        finally:
            if self.cap:
                self.cap.release()

    def _process_frame(self, frame, fps, width, height):
        """Process a single frame - detection logic handled by individual detector classes"""
        try:
            # Frame processing is now handled by individual detector classes
            # This method is kept for compatibility but does minimal processing
            pass
                    
        except Exception as e:
            logger.error(f"Error processing frame for camera {self.camera.id}: {str(e)}")
    
        
    def _update_camera_status(self, status):
        """Update camera status in database"""
        try:
            Camera.objects.filter(id=self.camera.id).update(
                status=status,
                last_online=timezone.now() if status == 'online' else None,
                updated_at=timezone.now()
            )
        except Exception as e:
            logger.error(f"Error updating camera status: {str(e)}")


class VideoFileProcessor:
    """
    Processor for video files - detection handled by individual detector classes
    """
    
    def __init__(self, video_file, app, device, manager):
        self.video_file = video_file
        self.video_name = os.path.basename(video_file)
        self.app = app
        self.device = device
        self.manager = manager
        
        self.is_running = False
        self.thread = None
        self.cap = None
        self.frame_count = 0
        
    def start(self):
        """Start processing this video file"""
        if self.is_running:
            return
        self.is_running = True
        try:
            # Detection is now handled by individual detector classes
            # This method is simplified to basic file processing
            logger.info(f"Started processing video file: {self.video_name}")
        except KeyboardInterrupt:
            print("\nShutting down...")

    def stop(self):
       """Stop processing this video file"""
       self.is_running = False
       if self.cap:
           self.cap.release()

# Global instance - replace the original detection manager
enhanced_detection_manager = EnhancedDetectionManager()