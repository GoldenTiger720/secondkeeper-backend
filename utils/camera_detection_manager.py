# utils/camera_detection_manager.py - Updated to integrate enhanced detection

import cv2
import numpy as np
import threading
import time
import os
import uuid
import logging
import queue
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from django.conf import settings
from django.utils import timezone
from django.db import transaction
import torch

from cameras.models import Camera
from alerts.models import Alert
from detectors import FireSmokeDetector, FallDetector, ViolenceDetector, ChokingDetector
from utils.model_manager import ModelManager
from utils.enhanced_video_processor import EnhancedVideoProcessor

logger = logging.getLogger('security_ai')

class CameraDetectionManager:
    """
    Main manager for handling automatic detection across all cameras with enhanced video priority
    """
    
    def __init__(self):
        self.active_cameras = {}  # camera_id -> CameraProcessor
        self.model_manager = ModelManager()
        self.video_processor = EnhancedVideoProcessor(self.model_manager)
        self.is_running = False
        self.main_thread = None
        self.stop_event = threading.Event()
        
        # Load all detectors
        self.detectors = {
            'fire_smoke': FireSmokeDetector(),
            'fall': FallDetector(),
            'violence': ViolenceDetector(),
            'choking': ChokingDetector()
        }
        
        # GPU optimization
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Detection settings
        self.frame_skip = 5  # Process every 5th frame for performance
        self.detection_cooldown = 30  # Seconds between detections for same camera
        self.video_clip_duration = 10  # Seconds for video clips
        
        # Alert tracking to prevent spam
        self.last_alerts = {}  # camera_id -> {alert_type: timestamp}
        
        # Test video directory check
        self.test_video_dir = os.path.join(settings.MEDIA_ROOT, 'testvideo')
        os.makedirs(self.test_video_dir, exist_ok=True)
        
    def start(self):
        """Start the detection manager with enhanced video file checking"""
        if self.is_running:
            logger.warning("Detection manager is already running")
            return
            
        # Check for enhanced detection manager first
        if self._should_use_enhanced_detection():
            logger.info("Test videos detected, delegating to Enhanced Detection Manager")
            try:
                from utils.enhanced_detection_manager import enhanced_detection_manager
                enhanced_detection_manager.start()
                return
            except ImportError:
                logger.warning("Enhanced detection manager not available, continuing with standard detection")
        
        logger.info("Starting Standard Camera Detection Manager")
        self.is_running = True
        self.stop_event.clear()
        
        # Start main processing thread
        self.main_thread = threading.Thread(target=self._main_loop, daemon=True)
        self.main_thread.start()
        
    def _should_use_enhanced_detection(self):
        """Check if enhanced detection should be used (if test videos exist)"""
        try:
            import glob
            video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv', '*.webm']
            
            for extension in video_extensions:
                pattern = os.path.join(self.test_video_dir, extension)
                if glob.glob(pattern):
                    return True
            return False
        except Exception as e:
            logger.error(f"Error checking for test videos: {str(e)}")
            return False
        
    def stop(self):
        """Stop the detection manager"""
        if not self.is_running:
            return
            
        logger.info("Stopping Camera Detection Manager")
        self.is_running = False
        self.stop_event.set()
        
        # Stop all camera processors
        for processor in self.active_cameras.values():
            processor.stop()
            
        # Wait for main thread to finish
        if self.main_thread and self.main_thread.is_alive():
            self.main_thread.join(timeout=10)
            
        self.active_cameras.clear()
        
    def _main_loop(self):
        """Main processing loop with periodic enhanced detection check"""
        while self.is_running and not self.stop_event.is_set():
            try:
                # Periodically check if we should switch to enhanced detection
                if self._should_use_enhanced_detection():
                    logger.info("Test videos detected during runtime, switching to Enhanced Detection Manager")
                    try:
                        from utils.enhanced_detection_manager import enhanced_detection_manager
                        # Stop current processing
                        self.stop()
                        # Start enhanced detection
                        enhanced_detection_manager.start()
                        return
                    except ImportError:
                        logger.warning("Enhanced detection manager not available")
                
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
                
                # Clean up old alerts from tracking
                self._cleanup_old_alerts()
                
            except Exception as e:
                logger.error(f"Error in main detection loop: {str(e)}")
            
            # Sleep for a bit before next iteration
            time.sleep(10)
            
    def _start_camera_processor(self, camera):
        """Start a processor for a single camera"""
        try:
            camera_id = str(camera.id)
            logger.info(f"Starting processor for camera {camera_id} - {camera.name}")
            
            processor = CameraProcessor(
                camera, self.detectors, self.device, self, self.video_processor
            )
            processor.start()
            
            self.active_cameras[camera_id] = processor
            
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
            
    def _cleanup_old_alerts(self):
        """Clean up old alert timestamps"""
        current_time = time.time()
        cutoff_time = current_time - self.detection_cooldown
        
        for camera_id in list(self.last_alerts.keys()):
            camera_alerts = self.last_alerts[camera_id]
            for alert_type in list(camera_alerts.keys()):
                if camera_alerts[alert_type] < cutoff_time:
                    del camera_alerts[alert_type]
            
            # Remove camera entry if no alerts
            if not camera_alerts:
                del self.last_alerts[camera_id]
                
    def should_create_alert(self, camera_id, alert_type):
        """Check if we should create a new alert (to prevent spam)"""
        current_time = time.time()
        
        if camera_id not in self.last_alerts:
            self.last_alerts[camera_id] = {}
            
        last_alert_time = self.last_alerts[camera_id].get(alert_type, 0)
        
        if current_time - last_alert_time >= self.detection_cooldown:
            self.last_alerts[camera_id][alert_type] = current_time
            return True
            
        return False
        
    def create_pending_alert(self, camera, alert_type, confidence, frame, detection_results):
        """Create alert for reviewer confirmation instead of direct notification"""
        try:
            # Use the enhanced video processor to create alert with video
            alert = self.video_processor.process_detection_with_video(
                camera, alert_type, confidence, detection_results
            )
            
            if alert:
                logger.info(f"Created pending alert {alert.id} for {alert_type} detection on camera {camera.id}")
                return alert
            else:
                logger.error(f"Failed to create pending alert for {alert_type} detection")
                return None
                
        except Exception as e:
            logger.error(f"Error creating pending alert: {str(e)}")
            return None


class CameraProcessor:
    """
    Processor for a single camera with enhanced detection
    """
    
    def __init__(self, camera, detectors, device, manager, video_processor):
        self.camera = camera
        self.detectors = detectors
        self.device = device
        self.manager = manager
        self.video_processor = video_processor
        
        self.is_running = False
        self.thread = None
        self.cap = None
        self.frame_count = 0
        
        # Get detection settings per camera using updated thresholds
        config = manager.model_manager.get_detector_config('fire_smoke')  # Default config
        self.confidence_threshold = camera.confidence_threshold or config['conf_threshold']
        self.iou_threshold = camera.iou_threshold or config['iou_threshold']
        self.image_size = camera.image_size or config['image_size']
        
    def start(self):
        """Start processing this camera"""
        if self.is_running:
            return
            
        self.is_running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        """Stop processing this camera"""
        self.is_running = False
        if self.cap:
            self.cap.release()
            
    def update_camera_data(self, camera):
        """Update camera data"""
        self.camera = camera
        # Update thresholds based on camera settings
        config = self.manager.model_manager.get_detector_config('fire_smoke')
        self.confidence_threshold = camera.confidence_threshold or config['conf_threshold']
        self.iou_threshold = camera.iou_threshold or config['iou_threshold']
        self.image_size = camera.image_size or config['image_size']
        
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
                
                # Skip frames for performance
                if self.frame_count % self.manager.frame_skip != 0:
                    continue
                    
                # Process frame with all enabled detectors
                self._process_frame(frame)
                
        except Exception as e:
            logger.error(f"Error in camera {self.camera.id} processing loop: {str(e)}")
            self._update_camera_status('error')
        finally:
            if self.cap:
                self.cap.release()

    def _process_frame(self, frame):
        """Process a single frame with conditional detection logic"""
        try:
            # First, check for people using person detector
            person_detector = self.detectors.get('person')
            if person_detector is None:
                # Add person detector if not available
                from detectors import PersonDetector
                person_detector = PersonDetector()
                self.detectors['person'] = person_detector
            
            # Get person detection configuration
            person_config = self.manager.model_manager.get_detector_config('person')
            person_conf_threshold = person_config['conf_threshold']
            person_iou_threshold = person_config['iou_threshold']
            person_image_size = person_config['image_size']
            
            # Run person detection
            person_annotated_frame, person_results = person_detector.predict_video_frame(
                frame, 
                person_conf_threshold,
                person_iou_threshold,
                person_image_size
            )
            
            # Check if person is detected
            has_person, person_confidence = self._check_detection_results(person_results, person_conf_threshold)
            
            detectors_to_run = []
            
            if has_person:
                # Person detected - check for fall, choking, violence
                if self.camera.fall_detection:
                    detectors_to_run.append('fall')
                if self.camera.violence_detection:
                    detectors_to_run.append('violence')
                if self.camera.choking_detection:
                    detectors_to_run.append('choking')
            else:
                # No person detected - check for fire/smoke
                if self.camera.fire_smoke_detection:
                    detectors_to_run.append('fire_smoke')
            
            # Run appropriate detectors based on person detection
            for detector_type in detectors_to_run:
                try:
                    detector = self.detectors[detector_type]
                    
                    # Get detector-specific configuration
                    config = self.manager.model_manager.get_detector_config(detector_type)
                    conf_threshold = config['conf_threshold']
                    iou_threshold = config['iou_threshold']
                    image_size = config['image_size']
                    
                    # Run detection
                    annotated_frame, results = detector.predict_video_frame(
                        frame, 
                        conf_threshold,
                        iou_threshold,
                        image_size
                    )
                    
                    # Check for detections
                    has_detection, max_confidence = self._check_detection_results(results, conf_threshold)
                    
                    if has_detection and max_confidence >= conf_threshold:
                        # Check if we should create an alert
                        if self.manager.should_create_alert(str(self.camera.id), detector_type):
                            # Create pending alert with bounding box video
                            self._create_detection_alert_with_bbox(
                                detector_type,
                                max_confidence,
                                frame,
                                results
                            )
                            
                except Exception as e:
                    logger.error(f"Error running {detector_type} detector on camera {self.camera.id}: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error processing frame for camera {self.camera.id}: {str(e)}")

    def _create_detection_alert_with_bbox(self, detector_type, confidence, frame, detection_results):
        """Create detection alert with bounding box video"""
        try:
            # Use the enhanced video processor to create alert with bounding box video
            alert = self.video_processor.process_detection_with_video_and_bbox(
                self.camera,
                detector_type,
                confidence,
                detection_results,
                frame
            )
            
            if alert:
                logger.info(f"Created detection alert {alert.id} with bounding box video for {detector_type} on camera {self.camera.id}")
                return alert
            else:
                logger.error(f"Failed to create detection alert with bounding box video for {detector_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating detection alert with bounding box video: {str(e)}")
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


# Global instance - this will automatically delegate to enhanced detection when needed
detection_manager = CameraDetectionManager()