# utils/enhanced_video_processor_patch.py - Simplified patch for video-only output

import os
import uuid
import cv2
import numpy as np
from datetime import datetime
from django.conf import settings
from django.utils import timezone
from django.db import transaction
from alerts.models import Alert

def add_test_detection_support(video_processor_class):
    """
    Add test detection support to the original EnhancedVideoProcessor class
    Focus on creating videos with bounding boxes, no metadata files
    """
    
    def create_test_detection_alert_with_bbox(self, camera, alert_type, confidence, detection_results, source_video_name, frame=None):
        """
        Create alert for test video detection with bounding box video only
        
        Args:
            camera: Camera model instance (test camera)
            alert_type: Type of detection
            confidence: Detection confidence
            detection_results: Detection results from model
            source_video_name: Name of the source video file
            frame: Current frame for processing
            
        Returns:
            Alert: Created alert instance or None if failed
        """
        try:
            with transaction.atomic():
                # Create output directory for test detections
                test_output_dir = os.path.join(settings.MEDIA_ROOT, 'test_detections')
                test_dir = os.path.join(test_output_dir, alert_type)
                os.makedirs(test_dir, exist_ok=True)
                
                # Create detection metadata
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                unique_id = uuid.uuid4().hex[:8]
                
                # Generate file names (video only)
                video_filename = f"test_{alert_type}_{timestamp}_{unique_id}.mp4"
                video_path = os.path.join(test_dir, video_filename)
                
                thumbnail_filename = f"test_{alert_type}_{timestamp}_{unique_id}_thumb.jpg"
                thumbnail_path = os.path.join(test_dir, thumbnail_filename)
                
                # Extract bounding boxes from detection results
                bboxes = self._extract_bounding_boxes_patch(detection_results)
                
                # Create video with bounding boxes from source video
                video_success = self._create_detection_video_from_source_patch(
                    source_video_name, video_path, bboxes, alert_type, confidence
                )
                
                if video_success:
                    # Create thumbnail from detection frame
                    if frame is not None:
                        annotated_frame = self._draw_bounding_boxes_patch(frame.copy(), bboxes, alert_type, confidence)
                        cv2.imwrite(thumbnail_path, annotated_frame)
                    else:
                        # Create thumbnail from first frame of output video
                        self._create_thumbnail_from_video_patch(video_path, thumbnail_path, bboxes, alert_type, confidence)
                    
                    # Determine severity
                    severity = self._determine_test_severity(alert_type, confidence)
                    
                    # Get relative paths for database storage
                    video_relative_path = os.path.relpath(video_path, settings.MEDIA_ROOT)
                    thumbnail_relative_path = os.path.relpath(thumbnail_path, settings.MEDIA_ROOT)
                    
                    # Create alert with video file (no metadata files)
                    alert = Alert.objects.create(
                        title=f"TEST {alert_type.replace('_', ' ').title()} Detection - {source_video_name}",
                        description=f"Test detection of {alert_type.replace('_', ' ')} from video file {source_video_name} with {confidence:.2f} confidence. Detection highlighted with bounding boxes.",
                        alert_type=alert_type,
                        severity=severity,
                        confidence=confidence,
                        camera=camera,
                        location=f"Test Video: {source_video_name}",
                        video_file=video_relative_path,
                        thumbnail=thumbnail_relative_path,
                        status='pending_review',
                        notes=f"Detected in test video: {source_video_name}. Unique ID: {unique_id}. Bounding boxes: {len(bboxes)}"
                    )
                    
                    import logging
                    logger = logging.getLogger('security_ai')
                    logger.info(f"Created test alert {alert.id} with detection video for {alert_type} in {source_video_name}")
                    return alert
                else:
                    import logging
                    logger = logging.getLogger('security_ai')
                    logger.error(f"Failed to create detection video for {alert_type}")
                    return None
                    
        except Exception as e:
            import logging
            logger = logging.getLogger('security_ai')
            logger.error(f"Error creating test detection alert with video: {str(e)}")
            return None
    
    def _create_detection_video_from_source_patch(self, source_video_name, output_video_path, bboxes, alert_type, confidence):
        """
        Create a detection video with bounding boxes from source video
        """
        try:
            # Get the full path to source video
            source_video_path = os.path.join(settings.MEDIA_ROOT, 'testvideo', source_video_name)
            
            if not os.path.exists(source_video_path):
                import logging
                logger = logging.getLogger('security_ai')
                logger.error(f"Source video not found: {source_video_path}")
                return False
            
            # Open source video
            cap = cv2.VideoCapture(source_video_path)
            if not cap.isOpened():
                import logging
                logger = logging.getLogger('security_ai')
                logger.error(f"Failed to open source video: {source_video_path}")
                return False
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            
            # Calculate frames for the clip (10 seconds max)
            max_frames = min(total_frames, fps * 10)
            
            import logging
            logger = logging.getLogger('security_ai')
            logger.info(f"Creating detection video: {max_frames} frames at {fps} FPS")
            
            frame_count = 0
            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Draw bounding boxes on frame
                annotated_frame = self._draw_bounding_boxes_patch(frame.copy(), bboxes, alert_type, confidence)
                
                # Write frame to output video
                out.write(annotated_frame)
                frame_count += 1
            
            # Release resources
            cap.release()
            out.release()
            
            logger.info(f"Successfully created detection video: {output_video_path}")
            return True
            
        except Exception as e:
            import logging
            logger = logging.getLogger('security_ai')
            logger.error(f"Error creating detection video: {str(e)}")
            return False
    
    def _extract_bounding_boxes_patch(self, detection_results):
        """Extract bounding boxes from YOLO detection results"""
        bboxes = []
        
        try:
            for r in detection_results:
                if r.boxes is not None and len(r.boxes) > 0:
                    boxes = r.boxes.xyxy.cpu().numpy()  # Get boxes in xyxy format
                    confidences = r.boxes.conf.cpu().numpy()
                    
                    if hasattr(r.boxes, 'cls'):
                        classes = r.boxes.cls.cpu().numpy()
                    else:
                        classes = [0] * len(boxes)  # Default class if not available
                    
                    for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                        x1, y1, x2, y2 = box.astype(int)
                        bboxes.append({
                            'x1': x1,
                            'y1': y1,
                            'x2': x2,
                            'y2': y2,
                            'confidence': float(conf),
                            'class': int(cls),
                            'class_name': r.names.get(int(cls), 'detection') if hasattr(r, 'names') else 'detection'
                        })
                        
        except Exception as e:
            import logging
            logger = logging.getLogger('security_ai')
            logger.error(f"Error extracting bounding boxes: {str(e)}")
            
        return bboxes
    
    def _draw_bounding_boxes_patch(self, frame, bboxes, alert_type, confidence):
        """Draw bounding boxes on frame with enhanced styling"""
        # Color mapping for different detection types
        color_map = {
            'fire_smoke': (0, 0, 255),    # Red
            'fall': (0, 255, 255),        # Yellow  
            'violence': (0, 165, 255),    # Orange
            'choking': (255, 0, 0),       # Blue
            'person': (0, 255, 0)         # Green
        }
       
        color = color_map.get(alert_type, (255, 255, 255))  # Default white
       
        # Draw title at top of frame
        title_text = f"{alert_type.replace('_', ' ').title()} Detection"
        cv2.putText(frame, title_text, (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
       
        # Draw overall confidence
        conf_text = f"Confidence: {confidence:.2f}"
        cv2.putText(frame, conf_text, (10, 70), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
       
        # Draw bounding boxes
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
            bbox_confidence = bbox['confidence']
            class_name = bbox['class_name']
           
            # Draw rectangle with thicker border
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
           
            # Draw inner rectangle for better visibility
            cv2.rectangle(frame, (x1+1, y1+1), (x2-1, y2-1), (255, 255, 255), 1)
           
            # Draw label with background
            label = f"{class_name}: {bbox_confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
           
            # Draw label background rectangle
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 15), 
                        (x1 + label_size[0] + 10, y1), color, -1)
           
            # Draw label text
            cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
       
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
       
        return frame
   
    def _create_thumbnail_from_video_patch(self, video_path, thumbnail_path, bboxes, alert_type, confidence):
        """Create thumbnail from first frame of video"""
        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    annotated_frame = self._draw_bounding_boxes_patch(frame, bboxes, alert_type, confidence)
                    cv2.imwrite(thumbnail_path, annotated_frame)
                cap.release()
        except Exception as e:
            import logging
            logger = logging.getLogger('security_ai')
            logger.error(f"Error creating thumbnail from video: {str(e)}")
    
    # Add methods to the class
    video_processor_class.create_test_detection_alert_with_bbox = create_test_detection_alert_with_bbox
    video_processor_class._create_detection_video_from_source_patch = _create_detection_video_from_source_patch
    video_processor_class._extract_bounding_boxes_patch = _extract_bounding_boxes_patch
    video_processor_class._draw_bounding_boxes_patch = _draw_bounding_boxes_patch
    video_processor_class._create_thumbnail_from_video_patch = _create_thumbnail_from_video_patch
    
    return video_processor_class

# Apply the patch to the original EnhancedVideoProcessor
from utils.enhanced_video_processor import EnhancedVideoProcessor
add_test_detection_support(EnhancedVideoProcessor)