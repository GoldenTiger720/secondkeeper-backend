import cv2
import numpy as np
import math
import logging
from ultralytics import YOLO
from .base_detector import BaseDetector
from django.conf import settings

logger = logging.getLogger("security_ai")


class FallDetector(BaseDetector):
    """Analyzes fall incidents based on pose keypoints"""

    def __init__(self):
        super().__init__()
        self.fall_angle_threshold = 45  # degrees from vertical
        self.ground_proximity_threshold = 0.15  # normalized distance
        model_path=settings.MODEL_PATHS['pose_model']
        self.model = YOLO(model_path)

    def calculate_body_angle(self, keypoints, confidences):
        """Calculate the angle of the body from vertical"""
        # Use shoulder to hip line to determine body orientation
        left_shoulder = keypoints[self.KEYPOINTS['left_shoulder']]
        right_shoulder = keypoints[self.KEYPOINTS['right_shoulder']]
        left_hip = keypoints[self.KEYPOINTS['left_hip']]
        right_hip = keypoints[self.KEYPOINTS['right_hip']]
        
        shoulder_conf = min(confidences[self.KEYPOINTS['left_shoulder']], 
                           confidences[self.KEYPOINTS['right_shoulder']])
        hip_conf = min(confidences[self.KEYPOINTS['left_hip']], 
                      confidences[self.KEYPOINTS['right_hip']])
        
        if shoulder_conf > self.CONFIDENCE_THRESHOLD and hip_conf > self.CONFIDENCE_THRESHOLD:
            shoulder_center = np.array([(left_shoulder[0] + right_shoulder[0]) / 2,
                                      (left_shoulder[1] + right_shoulder[1]) / 2])
            hip_center = np.array([(left_hip[0] + right_hip[0]) / 2,
                                 (left_hip[1] + right_hip[1]) / 2])
            
            # Calculate angle from vertical
            body_vector = hip_center - shoulder_center
            vertical_vector = np.array([0, 1])  # Pointing down
            
            cos_angle = np.dot(body_vector, vertical_vector) / (np.linalg.norm(body_vector) * np.linalg.norm(vertical_vector))
            angle = math.degrees(math.acos(np.clip(cos_angle, -1, 1)))
            
            return angle, True
        return 0, False

    def detect_ground_proximity(self, keypoints, confidences, image_shape):
        """Detect if person is close to ground level"""
        # Check if major body parts are in lower portion of image
        key_points = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
        ground_points = 0
        total_points = 0
        
        for point in key_points:
            idx = self.KEYPOINTS[point]
            if confidences[idx] > self.CONFIDENCE_THRESHOLD:
                y_position = keypoints[idx][1] / image_shape[0]  # Normalize to image height
                if y_position > (1 - self.ground_proximity_threshold):
                    ground_points += 1
                total_points += 1
        
        return (ground_points / total_points) > 0.5 if total_points > 0 else False

    def safe_tensor_to_numpy(self, tensor_data):
        """Safely convert tensor data to numpy arrays"""
        try:
            if isinstance(tensor_data, np.ndarray):
                return tensor_data
            if hasattr(tensor_data, 'cpu'):
                return tensor_data.cpu().numpy()
            if hasattr(tensor_data, 'numpy'):
                return tensor_data.numpy()
            return np.array(tensor_data)
        except Exception as e:
            logger.error(f"Error converting tensor to numpy: {e}")
            return np.array([])

    def analyze_keypoints(self, keypoints, confidences, image_shape):
        """Analyze keypoints for fall detection"""
        features = {
            'body_horizontal': False,
            'ground_proximity': False,
            'detection_state': "NORMAL",
            'detection_probability': 0.0,
            'body_angle': 0,
            'detection_type': 'fall'
        }

        # Calculate body angle
        body_angle, angle_detected = self.calculate_body_angle(keypoints, confidences)
        features['body_angle'] = body_angle
        
        if angle_detected and body_angle > self.fall_angle_threshold:
            features['body_horizontal'] = True

        # Check ground proximity
        features['ground_proximity'] = self.detect_ground_proximity(keypoints, confidences, image_shape)

        # Fall detection logic
        if features['body_horizontal'] and features['ground_proximity']:
            features['detection_state'] = "FALL DETECTED (GROUND)"
            features['detection_probability'] = 0.95
        elif features['body_horizontal']:
            features['detection_state'] = "FALL DETECTED (TILTED)"
            features['detection_probability'] = 0.7
        elif features['ground_proximity']:
            features['detection_state'] = "POSSIBLE FALL"
            features['detection_probability'] = 0.4
        else:
            features['detection_state'] = "NORMAL"
            features['detection_probability'] = 0.0

        return features

    def draw_annotations(self, frame, keypoints, confidences, features, confidence_threshold):
        """Draw fall-specific annotations on frame"""
        try:
            # Draw skeleton and keypoints
            frame = self.draw_skeleton(frame, keypoints, confidences, color=(0, 255, 0), thickness=2)
            frame = self.draw_keypoints(frame, keypoints, confidences, color=(0, 255, 0), radius=5)
            
            # Draw body angle indicator
            if 'body_angle' in features:
                angle_text = f"Body Angle: {features['body_angle']:.1f}°"
                cv2.putText(frame, angle_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Highlight if person is horizontal
            if features['body_horizontal']:
                cv2.putText(frame, "HORIZONTAL POSITION", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Highlight ground proximity
            if features['ground_proximity']:
                cv2.putText(frame, "GROUND PROXIMITY", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Draw status
            color = (0, 0, 255) if features['detection_probability'] > 0.7 else (0, 255, 0)
            # cv2.putText(frame, f"FALL: {features['detection_state']}", 
            #            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
        except Exception as e:
            logger.error(f"Error drawing fall annotations: {e}")
        
        return frame

    def analyze(self, frame, confidence, alert_threshold, finger_tips=None):
        try:
            results = self.model(frame, conf=confidence, verbose=False)
            annotated_frame = frame.copy()
            max_probability = 0.0
            detected = False
            
            for result in results:
                if result.keypoints is not None:
                    keypoints = self.safe_tensor_to_numpy(result.keypoints.xy)
                    confidences = self.safe_tensor_to_numpy(result.keypoints.conf)
                    
                    if keypoints.size == 0 or confidences.size == 0:
                        continue
                    
                    for kpts, confs in zip(keypoints, confidences):
                        features = self.analyze_keypoints(kpts, confs, frame.shape[:2])
                        max_probability = max(max_probability, features['detection_probability'])
                        
                        annotated_frame = self.draw_annotations(
                            annotated_frame, kpts, confs, features, confidence
                        )
                        
                        if features['detection_probability'] >= alert_threshold:
                            detected = True
            
            detection_status = {
                'type': 'fall',
                'detected': detected,
                'probability': max_probability,
                'state': 'FALL DETECTED' if detected else 'NORMAL'
            }
            
            return annotated_frame, detection_status
            
        except Exception as e:
            logger.error(f"Error in fall analysis: {e}")
            detection_status = {
                'type': 'fall',
                'detected': False,
                'probability': 0.0,
                'state': 'ERROR'
            }
            return frame, detection_status