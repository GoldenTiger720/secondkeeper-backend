# utils/enhanced_detection_manager.py - COMPLETE FIX with proper alert storage and video recording

import cv2
import numpy as np
import threading
import time
import os
import uuid
import logging
import queue
import glob
import torch
import multiprocessing
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from django.conf import settings
from django.utils import timezone
from django.db import transaction
from concurrent.futures import ThreadPoolExecutor
from cameras.models import Camera
from alerts.models import Alert
# from detectors.fire_detector_class import FireSmokeDetector
# from detectors.choking_detector_class import ChokingDetector
# from detectors.fall_detector_class import FallDetector
from detectors.violence_detector_class import ViolenceDetector
# from utils.enhanced_video_processor import EnhancedVideoProcessor

logger = logging.getLogger('security_ai')

class VideoRecordingManager:
    """
    Manages 15-second video recordings for detections
    """
    def __init__(self, video_processor=None):
        self.active_recordings = {}  # camera_id -> recording_info
        self.lock = threading.Lock()
        self.video_processor = video_processor
        
    def start_recording(self, camera, alert_type, confidence, detection_results, annotated_frame):
        """Start a 15-second video recording for a detection"""
        try:
            camera_id = str(camera.id)
            
            with self.lock:
                # Stop any existing recording for this camera
                if camera_id in self.active_recordings:
                    self._stop_recording(camera_id)
                
                # Create output directory
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                unique_id = uuid.uuid4().hex[:8]
                
                # Create detection-specific directory
                if camera.user.role == 'admin':
                    # For test videos
                    output_dir = os.path.join(settings.MEDIA_ROOT, 'test_detections', alert_type)
                else:
                    # For real cameras
                    output_dir = os.path.join(settings.MEDIA_ROOT, 'alert_videos', alert_type)
                
                os.makedirs(output_dir, exist_ok=True)
                
                # Generate file names
                video_filename = f"{alert_type}_{timestamp}_{unique_id}.mp4"
                video_path = os.path.join(output_dir, video_filename)
                
                thumbnail_filename = f"{alert_type}_{timestamp}_{unique_id}_thumb.jpg"
                thumbnail_path = os.path.join(output_dir, thumbnail_filename)
                
                # Save thumbnail from annotated frame
                cv2.imwrite(thumbnail_path, annotated_frame)
                
                # Start recording thread
                recording_info = {
                    'camera': camera,
                    'alert_type': alert_type,
                    'confidence': confidence,
                    'detection_results': detection_results,
                    'video_path': video_path,
                    'thumbnail_path': thumbnail_path,
                    'start_time': time.time(),
                    'frames_recorded': 0,
                    'stop_event': threading.Event(),
                    'video_writer': None,
                    'unique_id': unique_id
                }
                
                self.active_recordings[camera_id] = recording_info
                
                # Start recording thread
                recording_thread = threading.Thread(
                    target=self._record_video_thread,
                    args=(camera_id, recording_info),
                    daemon=True
                )
                recording_thread.start()
                
                return {
                    'video_path': video_path,
                    'thumbnail_path': thumbnail_path,
                    'unique_id': unique_id
                }
                
        except Exception as e:
            logger.error(f"Error starting video recording: {str(e)}")
            return None
    
    def _record_video_thread(self, camera_id, recording_info):
        """Thread function to record 15 seconds of video"""
        try:
            camera = recording_info['camera']
            video_path = recording_info['video_path']
            alert_type = recording_info['alert_type']
            confidence = recording_info['confidence']
            stop_event = recording_info['stop_event']
            
            # Open camera stream
            cap = cv2.VideoCapture(camera.stream_url)
            if not cap.isOpened():
                logger.error(f"Failed to open camera stream for recording: {camera.stream_url}")
                return
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            recording_info['video_writer'] = out
            
            # Record for 15 seconds
            target_frames = fps * 15  # 15 seconds
            frames_recorded = 0
            
            logger.info(f"Recording {target_frames} frames at {fps} FPS for {alert_type} detection")
            
            while frames_recorded < target_frames and not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame during recording")
                    break
                
                # Draw detection info on frame
                annotated_frame = self._annotate_detection_frame(
                    frame.copy(), alert_type, confidence, frames_recorded, target_frames
                )
                
                # Write frame to video
                out.write(annotated_frame)
                frames_recorded += 1
                recording_info['frames_recorded'] = frames_recorded
                
                # Small delay to maintain proper frame rate
                time.sleep(1.0 / fps)
            
            # Release resources
            cap.release()
            out.release()
            
            logger.info(f"Completed recording {frames_recorded} frames for {alert_type} detection")
            
            # Create alert in database after recording is complete
            self._create_alert_after_recording(recording_info)
            
        except Exception as e:
            logger.error(f"Error in video recording thread: {str(e)}")
        finally:
            # Clean up
            with self.lock:
                if camera_id in self.active_recordings:
                    del self.active_recordings[camera_id]
    
    def _annotate_detection_frame(self, frame, alert_type, confidence, frame_num, total_frames):
        """Add detection information to frame"""
        # Color mapping for different detection types
        color_map = {
            'fire_smoke': (0, 0, 255),    # Red
            'fall': (0, 255, 255),        # Yellow
            'violence': (0, 165, 255),    # Orange
            'choking': (255, 0, 0),       # Blue
        }
        
        color = color_map.get(alert_type, (255, 255, 255))
        
        # Draw detection info
        title_text = f"{alert_type.replace('_', ' ').title()} Detection - RECORDING"
        cv2.putText(frame, title_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        # Draw confidence
        conf_text = f"Confidence: {confidence:.2f}"
        cv2.putText(frame, conf_text, (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw recording progress
        progress = (frame_num / total_frames) * 100
        progress_text = f"Recording: {progress:.1f}% ({frame_num}/{total_frames})"
        cv2.putText(frame, progress_text, (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame
    
    def _create_alert_after_recording(self, recording_info):
        """Create alert in database after video recording is complete"""
        try:
            camera = recording_info['camera']
            alert_type = recording_info['alert_type']
            confidence = recording_info['confidence']
            video_path = recording_info['video_path']
            thumbnail_path = recording_info['thumbnail_path']
            unique_id = recording_info['unique_id']
            
            # Check if video file was created successfully
            if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
                logger.error(f"Video file not created or empty: {video_path}")
                return None
            
            # Convert video to web-compatible format if video processor is available
            final_video_path = video_path
            if self.video_processor:
                try:
                    logger.info(f"Converting video to web-compatible format: {video_path}")
                    converted_path = self.video_processor.convert_to_web_compatible(video_path)
                    
                    if converted_path != video_path and os.path.exists(converted_path):
                        # Conversion successful, remove original file
                        try:
                            os.remove(video_path)
                            final_video_path = converted_path
                            logger.info(f"Successfully converted video and removed original: {final_video_path}")
                        except Exception as e:
                            logger.warning(f"Could not remove original video file: {str(e)}")
                            final_video_path = converted_path
                    else:
                        logger.warning("Video conversion failed or returned same path, keeping original")
                        
                except Exception as e:
                    logger.error(f"Error during video conversion: {str(e)}")
            else:
                logger.warning("No video processor available for conversion")
            
            video_path = final_video_path
            
            # Determine severity based on alert type and confidence
            severity = self._determine_alert_severity(alert_type, confidence)
            
            # Get relative paths for database storage
            video_relative_path = os.path.relpath(video_path, settings.MEDIA_ROOT)
            thumbnail_relative_path = os.path.relpath(thumbnail_path, settings.MEDIA_ROOT)
            
            # Create title based on camera type
            if camera.user.role == 'admin':
                title = f"TEST {alert_type.replace('_', ' ').title()} Detection"
                description = f"Test detection of {alert_type.replace('_', ' ')} with {confidence:.2f} confidence. 15-second web-compatible video recorded with detection highlights."
            else:
                title = f"{alert_type.replace('_', ' ').title()} Detection Alert"
                description = f"Detected {alert_type.replace('_', ' ')} on camera {camera.name} with {confidence:.2f} confidence. 15-second web-compatible video recorded for review."
            
            # Create alert with database transaction
            with transaction.atomic():
                alert = Alert.objects.create(
                    title=title,
                    description=description,
                    alert_type=alert_type,
                    severity=severity,
                    confidence=confidence,
                    camera=camera,
                    location=camera.name,
                    video_file=video_relative_path,
                    thumbnail=thumbnail_relative_path,
                    status='pending_review',
                    notes=f"15-second web-compatible video recorded. Unique ID: {unique_id}. Frames: {recording_info['frames_recorded']}"
                )
                
                logger.info(f"Alert {alert.id} created successfully for {alert_type} detection on camera {camera.id}")
                return alert
                
        except Exception as e:
            logger.error(f"Error creating alert after recording: {str(e)}")
            return None
    
    def _determine_alert_severity(self, alert_type, confidence):
        """Determine alert severity based on type and confidence"""
        # Base severity on confidence
        if confidence >= 0.9:
            base_severity = 'critical'
        elif confidence >= 0.7:
            base_severity = 'high'
        elif confidence >= 0.5:
            base_severity = 'medium'
        else:
            base_severity = 'low'
        
        # Adjust based on detection type priority
        high_priority_types = ['fire_smoke', 'choking']
        if alert_type in high_priority_types:
            if base_severity == 'medium':
                return 'high'
            elif base_severity == 'low':
                return 'medium'
        
        return base_severity
    
    def _stop_recording(self, camera_id):
        """Stop recording for a specific camera"""
        if camera_id in self.active_recordings:
            recording_info = self.active_recordings[camera_id]
            recording_info['stop_event'].set()
            
            # Release video writer if exists
            if recording_info.get('video_writer'):
                recording_info['video_writer'].release()

class EnhancedDetectionManager:
    """
    Enhanced detection manager that prioritizes video files over camera streams
    with proper alert storage and 15-second video recording
    """
    
    def __init__(self):
        self.active_cameras = {}  # camera_id -> CameraProcessor
        self.active_video_processors = {}  # video_file -> VideoFileProcessor
        # self.video_processor = EnhancedVideoProcessor()
        # self.recording_manager = VideoRecordingManager(self.video_processor)
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
        self.detection_cooldown = 30  # Seconds between detections for same source
        self.enabled_detectors = ['fire', 'fall', 'choking', 'violence']
        
        # Alert tracking to prevent spam
        self.last_alerts = {}  # source_id -> {alert_type: timestamp}
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
        
    def display_frame(self, detector_config):
        detector_class = None
        cap = None
        
        try:
            detector_type = detector_config['detector_type']
            
            # if detector_type == 'fire':
            #     detector_class = FireSmokeDetector(
            #         detector_config['rtsp_url'],
            #         detector_config['video_file'],
            #         detector_config['use_rtsp']
            #     )
            # elif detector_type == 'fall':
            #     detector_class = FallDetector(
            #         detector_config['rtsp_url'],
            #         detector_config['video_file'],
            #         detector_config['use_rtsp']
            #     )
            # elif detector_type == 'choking':
            #     detector_class = ChokingDetector(
            #         detector_config['rtsp_url'],
            #         detector_config['video_file'],
            #         detector_config['use_rtsp']
            #     )
            if detector_type == 'violence':
                detector_class = ViolenceDetector()
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
            
            print(f"[INFO] {detector_type.capitalize()} Detector started - Video: {width}x{height} @ {fps}fps")
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    # If it's a video file, loop it
                    if not detector_config['use_rtsp']:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        print(f"[INFO] {detector_type.capitalize()} Detector: End of stream reached")
                        break

                # Process frame using the detector class
                frame = detector_class.process_frame(frame, fps)

                # Show the frame with OpenCV (with detector type in window name)
                window_name = f"{detector_type.capitalize()} Detection - PID:{multiprocessing.current_process().pid}"
                cv2.imshow(window_name, frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print(f"[INFO] {detector_type.capitalize()} Detector: Quit key pressed")
                    break
                    
                frame_count += 1
                if frame_count % 300 == 0:  # Log every 300 frames (10 seconds at 30fps)
                    print(f"[INFO] {detector_type.capitalize()} Detector processed {frame_count} frames")
                    
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
        enabled_detectors = ['violence']
        use_rtsp = False  # Video files do not use RTSP
        rtsp_url = ""  
        
        # Create detector configurations (don't initialize the detector objects here)
        detector_configs = []
        
        for detector_type in enabled_detectors:
            detector_config = {
                'detector_type': detector_type,
                'rtsp_url': rtsp_url,
                'video_file': self.video_file,
                'use_rtsp': use_rtsp
            }
            detector_configs.append(detector_config)
            print(f"[INFO] Added {detector_type.capitalize()} Detector to configuration")

        if not detector_configs:
            print("[ERROR] No detectors configured!")
            return

        # Create a process for each detector configuration
        processes = []
        for detector_config in detector_configs:
            process = multiprocessing.Process(
                target=self.display_frame, 
                args=(detector_config,),
                name=f"{detector_config['detector_type'].capitalize()}DetectorProcess"
            )
            processes.append(process)
            process.start()
            print(f"[INFO] Started {detector_config['detector_type'].capitalize()} Detector process (PID: {process.pid})")

        try:
            print(f"[INFO] Running {len(processes)} detector processes. Press Ctrl+C to stop or 'q' in any window.")
            # Wait for all processes to complete
            for process in processes:
                process.join()
        except KeyboardInterrupt:
            print("[INFO] Terminating all processes...")
            for process in processes:
                if process.is_alive():
                    print(f"[INFO] Terminating {process.name}...")
                    process.terminate()
            
            # Wait for processes to terminate gracefully
            for process in processes:
                process.join(timeout=5)
                if process.is_alive():
                    print(f"[WARNING] Force killing {process.name}...")
                    process.kill()
                    process.join()
        
        print("[INFO] All detector processes have been stopped.")
        
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
                
                # Clean up old alerts from tracking
                self._cleanup_old_alerts()
                
            except Exception as e:
                logger.error(f"Error in main enhanced detection loop: {str(e)}")
            
            # Sleep for a bit before next iteration
            time.sleep(5)
    
    def should_create_alert(self, source_id, alert_type):
        """Check if we should create a new alert (to prevent spam)"""
        current_time = time.time()
        
        if source_id not in self.last_alerts:
            self.last_alerts[source_id] = {}
            
        last_alert_time = self.last_alerts[source_id].get(alert_type, 0)
        
        if current_time - last_alert_time >= self.detection_cooldown:
            self.last_alerts[source_id][alert_type] = current_time
            return True
            
        return False
    
    def create_alert_with_recording(self, camera, alert_type, confidence, detection_results, annotated_frame):
        """Create alert with 15-second video recording"""
        try:
            # Start 15-second video recording
            recording_result = self.recording_manager.start_recording(
                camera, alert_type, confidence, detection_results, annotated_frame
            )
            
            if recording_result:
                logger.info(f"Started 15-second recording for {alert_type} detection on camera {camera.id}")
                return recording_result
            else:
                logger.error(f"Failed to start recording for {alert_type} detection")
                return None
                
        except Exception as e:
            logger.error(f"Error creating alert with recording: {str(e)}")
            return None

    # ... [Include all other methods from the original file: _get_test_video_files, _process_video_files, etc.]
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
            enabled_detectors = ['violence']
            use_rtsp = False  # Video files do not use RTSP
            rtsp_url = ""
            self.video_file = video_file  # Set the current video file             
            self.run_detectors()
            
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
        # self.app = MultiDetectionSystem()
        try:
            camera_id = str(camera.id)
            logger.info(f"Starting processor for camera {camera_id} - {camera.name}")
            # processor = EnhancedCameraProcessor(
            #     camera, self.app, self.device, self, self.video_processor, self.recording_manager)
            # processor.start()
            
            # self.active_cameras[camera_id] = processor
            
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
            
    def _cleanup_old_alerts(self):
        """Clean up old alert timestamps"""
        current_time = time.time()
        cutoff_time = current_time - self.detection_cooldown
        
        for source_id in list(self.last_alerts.keys()):
            source_alerts = self.last_alerts[source_id]
            for alert_type in list(source_alerts.keys()):
                if source_alerts[alert_type] < cutoff_time:
                    del source_alerts[alert_type]
            
            # Remove source entry if no alerts
            if not source_alerts:
                del self.last_alerts[source_id]


class EnhancedCameraProcessor:
    """
    Enhanced processor for camera streams with proper alert creation and video recording
    """
    
    def __init__(self, camera, app, device, manager, video_processor, recording_manager):
        self.camera = camera
        self.app = app
        self.device = device
        self.manager = manager
        self.video_processor = video_processor
        self.recording_manager = recording_manager
        self.output_dir = os.path.join(settings.MEDIA_ROOT, 'alert_videos', 'choking')
        
        self.is_running = False
        self.thread = None
        self.cap = None
        self.frame_count = 0
        
        # Get detection settings per camera
        # config = self.manager.model_manager.get_detector_config('fire_smoke')
        # self.confidence_threshold = camera.confidence_threshold or config['conf_threshold']
        # self.iou_threshold = camera.iou_threshold or config['iou_threshold']
        # self.image_size = camera.image_size or config['image_size']
        
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
        # config = self.manager.model_manager.get_detector_config('fire_smoke')
        # self.confidence_threshold = camera.confidence_threshold or config['conf_threshold']
        # self.iou_threshold = camera.iou_threshold or config['iou_threshold']
        # self.image_size = camera.image_size or config['image_size']
        
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
        """Process a single frame with separated detection types to prevent false positives"""
        try:
            import concurrent.futures
            print(f"[INFO] Processing frame for camera {self.camera.id} - {self.camera.name} at {fps} FPS")
            
            # # CRITICAL FIX: Separate fire/smoke detection from pose-based detections
            # # Step 1: Check for fire/smoke first (non-person detection)
            # fire_smoke_detected = False
            # fire_smoke_confidence = 0.0
            
            # try:
            #     fire_result = self._run_standard_detection('fire_smoke', frame)
            #     if fire_result:
            #         has_fire_detection, fire_confidence, fire_annotated_frame, fire_results = fire_result
            #         if has_fire_detection:
            #             fire_smoke_detected = True
            #             fire_smoke_confidence = fire_confidence
                        
            #             # Create fire/smoke alert immediately
            #             source_id = f"camera_{self.camera.id}"
            #             if self.manager.should_create_alert(source_id, 'fire_smoke'):
            #                 logger.info(f"FIRE/SMOKE detected with confidence {fire_confidence:.2f} - Creating alert")
            #                 recording_result = self.manager.create_alert_with_recording(
            #                     self.camera,
            #                     'fire_smoke',
            #                     fire_confidence,
            #                     fire_results,
            #                     fire_annotated_frame
            #                 )
                            
            #                 if recording_result:
            #                     logger.info(f"Successfully started recording for fire/smoke detection")
            #                 else:
            #                     logger.error(f"Failed to start recording for fire/smoke detection")
            # except Exception as e:
            #     logger.error(f"Error in fire/smoke detection: {str(e)}")
            
            # # Step 2: ONLY run pose-based detections if NO fire/smoke was detected
            # # This prevents false positives where people near fire are misclassified as choking/falling
            # if not fire_smoke_detected:
            #     logger.debug("No fire/smoke detected - proceeding with pose-based detection")
                
            #     # Create thread pool for pose-based detections only
            #     with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            #         # Submit pose-based detection tasks only
            #         futures = {
            #             executor.submit(self._run_choking_detection, frame): 'choking',
            #             executor.submit(self._run_standard_detection, 'fall', frame): 'fall',
            #             executor.submit(self._run_standard_detection, 'violence', frame): 'violence'
            #         }
                    
            #         # Process pose-based results
            #         for future in concurrent.futures.as_completed(futures):
            #             detector_type = futures[future]
            #             try:
            #                 result = future.result()
            #                 if result:
            #                     has_detection, max_confidence, annotated_frame, results = result
                                
            #                     if has_detection:
            #                         # Check if we should create an alert
            #                         source_id = f"camera_{self.camera.id}"
            #                         if self.manager.should_create_alert(source_id, detector_type):
            #                             logger.info(f"Creating alert with recording for {detector_type} detection on camera {self.camera.id}")
            #                             recording_result = self.manager.create_alert_with_recording(
            #                                 self.camera,
            #                                 detector_type,
            #                                 max_confidence,
            #                                 results,
            #                                 annotated_frame
            #                             )
                                        
            #                             if recording_result:
            #                                 logger.info(f"Successfully started recording for {detector_type} detection")
            #                             else:
            #                                 logger.error(f"Failed to start recording for {detector_type} detection")
                        
            #             except Exception as e:
            #                 logger.error(f"Error in {detector_type} detection thread: {str(e)}")
            # else:
            #     logger.info(f"Fire/smoke detected with confidence {fire_smoke_confidence:.2f} - SKIPPING pose-based detections to prevent false positives")
                    
        except Exception as e:
            logger.error(f"Error processing frame for camera {self.camera.id}: {str(e)}")
    
    def _run_choking_detection(self, frame):
        """Run choking detection in separate thread using new detector"""
        try:
            # Use new choking detector's predict_video_frame method
            annotated_frame, results = self.choking_detector.predict_video_frame(
                'choking', frame, 0.5, 0.5, 640
            )
            
            # Check for detections in results
            has_detection, max_confidence = self._check_detection_results_choking(results, annotated_frame, frame)
            
            return has_detection, max_confidence, annotated_frame, results
                
        except Exception as e:
            logger.error(f"Error running choking detector: {str(e)}")
            return False, 0.0, frame, []
    
    def _run_standard_detection(self, detector_type, frame):
        """Run standard YOLO-based detection in separate thread"""
        try:
            detector = self.detectors[detector_type]
            
            # Get detector-specific configuration
            config = self.manager.model_manager.get_detector_config(detector_type)
            conf_threshold = config['conf_threshold']
            iou_threshold = config['iou_threshold']
            image_size = config['image_size']
            
            # Run detection
            annotated_frame, results = detector.predict_video_frame(
                detector_type,
                frame, 
                conf_threshold,
                iou_threshold,
                image_size
            )

            # Check for detections
            has_detection, max_confidence = self._check_detection_results(results, conf_threshold)
            
            return has_detection, max_confidence, annotated_frame, results
            
        except Exception as e:
            logger.error(f"Error running {detector_type} detector: {str(e)}")
            return False, 0.0, frame, []

    def _save_detection_video(self, alert_type, confidence, detection_results, annotated_frame, fps, width, height):
        try:
            print(self.output_dir)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_filename = f"{alert_type}_{timestamp}_{self.camera.id}.mp4"
            video_path = os.path.join(self.output_dir, video_filename)
            thumbnail_filename = f"{alert_type}_{timestamp}_{self.camera.id}_thumb.jpg"
            thumbnail_path = os.path.join(self.output_dir, thumbnail_filename)
            cv2.imwrite(thumbnail_path, annotated_frame)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

            frames_to_record = fps * 15  # 15 seconds
            frames_recorded = 0

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, self.frame_count - 1))
            while frames_recorded < frames_to_record and self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    break
                annotated = self._annotate_detection_frame(
                    frame.copy(), alert_type, confidence, frames_recorded, frames_to_record
                )
                out.write(annotated)
                frames_recorded += 1
                time.sleep(1.0 / fps)
            out.release()

            # Save alert to DB (as a test camera/user)
            self._create_detection_alert(alert_type, confidence, detection_results, video_path, thumbnail_path, frames_recorded, self.camera.id)
        except Exception as e:
            logger.error(f"Error saving CCD detection video: {str(e)}")

    def _annotate_detection_frame(self, frame, alert_type, confidence, frame_num, total_frames):
        color_map = {
            'fire_smoke': (0, 0, 255),
            'fall': (0, 255, 255),
            'violence': (0, 165, 255),
            'choking': (255, 0, 0),
        }
        color = color_map.get(alert_type, (255, 255, 255))
        title_text = f"{alert_type.replace('_', ' ').title()} Detection - {self.camera.id}"
        cv2.putText(frame, title_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        conf_text = f"Confidence: {confidence:.2f}"
        cv2.putText(frame, conf_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        progress = (frame_num / total_frames) * 100
        progress_text = f"Recording: {progress:.1f}% ({frame_num}/{total_frames})"
        cv2.putText(frame, progress_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return frame

    def _create_detection_alert(self, alert_type, confidence, detection_results, video_path, thumbnail_path, frames_recorded, unique_id):
        try:
            from django.contrib.auth import get_user_model
            User = get_user_model()
            admin_user = User.objects.filter(role='admin', is_active=True).first()
            if not admin_user:
                print(f"No admin user found for {self.camera.name} camera alerts")
                return None

            video_relative_path = os.path.relpath(video_path, settings.MEDIA_ROOT)
            thumbnail_relative_path = os.path.relpath(thumbnail_path, settings.MEDIA_ROOT)
            title = f"{self.camera.name} {alert_type.replace('_', ' ').title()} Detection"
            description = f"{self.camera.name} detection of {alert_type.replace('_', ' ')} with {confidence:.2f} confidence. 15-second video recorded."
            with transaction.atomic():
                alert = Alert.objects.create(
                    title=title,
                    description=description,
                    alert_type=alert_type,
                    severity='medium',
                    confidence=confidence,
                    camera=self.camera,
                    location=self.camera.name,
                    video_file=video_relative_path,
                    thumbnail=thumbnail_relative_path,
                    status='pending_review',
                    notes=f"{self.camera.name} 15-second video. Unique ID: {self.camera.id}. Frames: {frames_recorded}"
                )
                logger.info(f"CCD Alert {alert.id} created successfully for {alert_type}")
                return alert
        except Exception as e:
            logger.error(f"Error creating CCD alert: {str(e)}")
            return None

    def _check_detection_results(self, results, conf_threshold):
        """Check detection results for valid detections"""
        has_detection = False
        max_confidence = 0.0
        
        try:
            for r in results:
                if r.boxes is not None and len(r.boxes) > 0:
                    confidences = r.boxes.conf.tolist()
                    if confidences:
                        max_conf = max(confidences)
                        if max_conf >= conf_threshold:
                            has_detection = True
                            max_confidence = max(max_confidence, max_conf)
                            
        except Exception as e:
            logger.error(f"Error checking detection results: {str(e)}")
            
        return has_detection, max_confidence
    
    def _check_detection_results_choking(self, results, annotated_frame, original_frame):
        """Check choking detection results from pose-based analysis"""
        has_detection = False
        max_confidence = 0.0
        
        try:
            # For pose-based detectors, check if annotations were added to the frame
            # Compare annotated frame with original to detect changes
            if not np.array_equal(annotated_frame, original_frame):
                # Some annotation was added, indicating detection
                has_detection = True
                max_confidence = 0.85  # Default confidence for pose-based detection
            
            # Also check if keypoints are present in results
            for r in results:
                if r.keypoints is not None and hasattr(r.keypoints, 'xy') and r.keypoints.xy is not None:
                    # Keypoints detected, potential for choking detection
                    max_confidence = max(max_confidence, 0.7)
                    
        except Exception as e:
            logger.error(f"Error checking choking detection results: {str(e)}")
            
        return has_detection, max_confidence
        
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
    Processor for video files with detection capabilities and alert creation
    """
    
    def __init__(self, video_file, app, device, manager, video_processor):
        self.video_file = video_file
        self.video_name = os.path.basename(video_file)
        self.app = app
        self.device = device
        self.manager = manager
        self.video_processor = video_processor
        
        self.is_running = False
        self.thread = None
        self.cap = None
        self.frame_count = 0
        
    def start(self):
        """Start processing this video file"""
        if self.is_running:
            return
        self.is_running = True
        # self.thread = threading.Thread(target=self._process_loop, daemon=True)
        # self.thread.start()
        try:
            self.app.run_detection_threads(self.video_file)
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.app.cleanup()

    def stop(self):
       """Stop processing this video file"""
       self.is_running = False
       if self.cap:
           self.cap.release()
           
    def _process_loop(self):
        """Main processing loop for this video file"""
        try:
            # Open video file
            self.cap = cv2.VideoCapture(self.video_file)
            if not self.cap.isOpened():
                print(f"Failed to open video file: {self.video_file}")
                return
                
            print(f"Started processing video file: {self.video_name}")
            
            # Get video properties
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            logger.info(f"Video properties - Total frames: {total_frames}, FPS: {fps}, Duration: {duration:.2f}s")

            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    logger.info(f"Finished processing video file: {self.video_name}")
                    break
                    
                self.frame_count += 1
                
                # Skip frames for performance
                if self.frame_count % self.manager.frame_skip != 0:
                    continue
                    
                # Process frame with all detectors
                self._process_frame(frame)
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error in video file {self.video_name} processing loop: {str(e)}")
        finally:
            if self.cap:
                self.cap.release()
                
    def _process_frame(self, frame):
        """Process a single frame with separated detection types to prevent false positives"""
        try:
            import concurrent.futures
            
            # CRITICAL FIX: Separate fire/smoke detection from pose-based detections
            # Step 1: Check for fire/smoke first (non-person detection)
            fire_smoke_detected = False
            fire_smoke_confidence = 0.0
            
            try:
                fire_result = self._run_video_standard_detection('fire_smoke', frame)
                if fire_result:
                    has_fire_detection, fire_confidence, fire_annotated_frame, fire_results = fire_result
                    if has_fire_detection:
                        fire_smoke_detected = True
                        fire_smoke_confidence = fire_confidence
                        
                        # Create fire/smoke alert immediately
                        source_id = f"video_{self.video_name}"
                        if self.manager.should_create_alert(source_id, 'fire_smoke'):
                            logger.info(f"FIRE/SMOKE detected in video {self.video_name} with confidence {fire_confidence:.2f} - Creating alert")
                            self._create_test_alert(
                                'fire_smoke',
                                fire_confidence,
                                fire_results,
                                fire_annotated_frame
                            )
            except Exception as e:
                logger.error(f"Error in fire/smoke detection for video: {str(e)}")
            
            # Step 2: ONLY run pose-based detections if NO fire/smoke was detected
            # This prevents false positives where people near fire are misclassified as choking/falling
            if not fire_smoke_detected:
                logger.debug(f"No fire/smoke detected in video {self.video_name} - proceeding with pose-based detection")
                
                # Create thread pool for pose-based detections only
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    # Submit pose-based detection tasks only
                    futures = {
                        executor.submit(self._run_video_choking_detection, frame): 'choking',
                        executor.submit(self._run_video_standard_detection, 'fall', frame): 'fall',
                        executor.submit(self._run_video_standard_detection, 'violence', frame): 'violence'
                    }
                    
                    # Process pose-based results
                    for future in concurrent.futures.as_completed(futures):
                        detector_type = futures[future]
                        try:
                            result = future.result()
                            if result:
                                has_detection, max_confidence, annotated_frame, results = result
                                
                                if has_detection:
                                    # Check if we should create an alert
                                    source_id = f"video_{self.video_name}"
                                    if self.manager.should_create_alert(source_id, detector_type):
                                        # Create test alert with proper database storage
                                        self._create_test_alert(
                                            detector_type,
                                            max_confidence,
                                            results,
                                            annotated_frame
                                        )
                        
                        except Exception as e:
                            logger.error(f"Error in {detector_type} detection thread for video {self.video_name}: {str(e)}")
            else:
                logger.info(f"Fire/smoke detected in video {self.video_name} with confidence {fire_smoke_confidence:.2f} - SKIPPING pose-based detections to prevent false positives")
                
        except Exception as e:
            logger.error(f"Error processing frame for video {self.video_name}: {str(e)}")
    
    def _run_video_choking_detection(self, frame):
        """Run choking detection in separate thread for video files"""
        try:
            # Use the enhanced choking detector from the main detectors dict
            detector = self.manager.detectors.get('choking')
            if detector:
                annotated_frame, results = detector.predict_video_frame(
                    'choking', frame, 0.5, 0.5, 640
                )
                
                # Check for detections
                has_detection, max_confidence = self._check_detection_results_choking(results, annotated_frame, frame)
                
                return has_detection, max_confidence, annotated_frame, results
            else:
                return False, 0.0, frame, []
                
        except Exception as e:
            logger.error(f"Error running choking detector on video: {str(e)}")
            return False, 0.0, frame, []
    
    def _run_video_standard_detection(self, detector_type, frame):
        """Run standard YOLO-based detection in separate thread for video files"""
        try:
            detector = self.detectors[detector_type]
            
            # Get detector-specific configuration
            config = self.manager.model_manager.get_detector_config(detector_type)
            conf_threshold = config['conf_threshold']
            iou_threshold = config['iou_threshold']
            image_size = config['image_size']
            
            # Run detection
            annotated_frame, results = detector.predict_video_frame(
                detector_type,
                frame, 
                conf_threshold,
                iou_threshold,
                image_size
            )
            
            # Check for detections
            has_detection, max_confidence = self._check_detection_results(results, conf_threshold)
            
            return has_detection, max_confidence, annotated_frame, results
            
        except Exception as e:
            logger.error(f"Error running {detector_type} detector on video: {str(e)}")
            return False, 0.0, frame, []
            
    def _create_test_alert(self, alert_type, confidence, detection_results, annotated_frame):
        """Create test alert with proper database storage"""
        try:
            # Create a test camera for video files
            from django.contrib.auth import get_user_model
            User = get_user_model()
            
            # Get or create admin user for test videos
            admin_user = User.objects.filter(role='admin', is_active=True).first()
            if not admin_user:
                logger.warning("No admin user found for test video alerts")
                return None
            
            # Create or get test camera
            test_camera, created = Camera.objects.get_or_create(
                name=f"Test Video Camera - {self.video_name}",
                user=admin_user,
                defaults={
                    'stream_url': f"file://{self.video_name}",
                    'status': 'online',
                    'detection_enabled': True,
                    'fire_smoke_detection': True,
                    'fall_detection': True,
                    'violence_detection': True,
                    'choking_detection': True
                }
            )
            
            # Create alert using the video processor with enhanced features
            alert = self.video_processor.create_test_detection_alert_with_bbox(
                test_camera, 
                alert_type, 
                confidence, 
                detection_results, 
                self.video_name, 
                annotated_frame
            )
            
            if alert:
                logger.info(f"Created test alert {alert.id} for {alert_type} detection in {self.video_name}")
                return alert
            else:
                logger.error(f"Failed to create test alert for {alert_type} detection")
                return None
                
        except Exception as e:
            logger.error(f"Error creating test alert: {str(e)}")
            return None
    
    def _check_detection_results(self, results, conf_threshold):
        """Check detection results for valid detections"""
        has_detection = False
        max_confidence = 0.0
        
        try:
            for r in results:
                if r.boxes is not None and len(r.boxes) > 0:
                    confidences = r.boxes.conf.tolist()
                    if confidences:
                        max_conf = max(confidences)
                        if max_conf >= conf_threshold:
                            has_detection = True
                            max_confidence = max(max_confidence, max_conf)
                            
        except Exception as e:
            logger.error(f"Error checking detection results: {str(e)}")
            
        return has_detection, max_confidence
    
    def _check_detection_results_choking(self, results, annotated_frame, original_frame):
        """Check choking detection results from pose-based analysis for video"""
        has_detection = False
        max_confidence = 0.0
        
        try:
            # For pose-based detectors, check if annotations were added to the frame
            if not np.array_equal(annotated_frame, original_frame):
                has_detection = True
                max_confidence = 0.85
            
            # Also check if keypoints are present in results
            for r in results:
                if r.keypoints is not None and hasattr(r.keypoints, 'xy') and r.keypoints.xy is not None:
                    max_confidence = max(max_confidence, 0.7)
                    
        except Exception as e:
            logger.error(f"Error checking choking detection results: {str(e)}")
            
        return has_detection, max_confidence




# Global instance - replace the original detection manager
enhanced_detection_manager = EnhancedDetectionManager()