import cv2
import numpy as np
import math
import logging
from ultralytics import YOLO
import mediapipe as mp
from .base_detector import BaseDetector
from django.conf import settings

logger = logging.getLogger("security_ai")


class ChokingDetector(BaseDetector):
    """Analyzes choking signs based on pose keypoints and features"""

    def __init__(self):
        super().__init__()
        self.WRIST_NECK_THRESHOLD = 0.1
        self.ELBOW_NECK_THRESHOLD = 0.12
        self.WRIST_DISTANCE_THRESHOLD = 0.03  # Maximum distance between wrists when choking
        self.HAND_SYMMETRY_THRESHOLD = 0.05  # Threshold for hand symmetry detection
        self.ARM_NECK_COSINE_THRESHOLD = 0.8  # Cosine similarity threshold for arm direction
        self.SIDE_POSE_THRESHOLD = 0.15  # Threshold for detecting side pose
        model_path = settings.MODEL_PATHS['pose_model']
        self.model = YOLO(model_path)
        
        # Initialize MediaPipe for enhanced detection
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
        

    def calculate_wrist_to_neck_distance(self, wrist_point, neck_center, image_shape):
        """Calculate normalized distance between wrist and neck"""
        if neck_center is None:
            return float('inf')
        distance = np.linalg.norm(np.array(wrist_point) - np.array(neck_center))
        image_diagonal = math.sqrt(image_shape[0] ** 2 + image_shape[1] ** 2)
        return distance / image_diagonal

    def calculate_wrist_distance(self, left_wrist, right_wrist, image_shape):
        """Calculate normalized distance between both wrists"""
        distance = np.linalg.norm(np.array(left_wrist) - np.array(right_wrist))
        image_diagonal = math.sqrt(image_shape[0] ** 2 + image_shape[1] ** 2)
        return distance / image_diagonal

    def check_hand_symmetry(self, left_wrist, right_wrist, neck_center, image_shape):
        """Check if both hands are positioned symmetrically around the neck"""
        if neck_center is None:
            return False
        
        # Calculate distances from each wrist to neck
        left_to_neck = np.linalg.norm(np.array(left_wrist) - np.array(neck_center))
        right_to_neck = np.linalg.norm(np.array(right_wrist) - np.array(neck_center))
        
        # Calculate difference in distances (normalized)
        image_diagonal = math.sqrt(image_shape[0] ** 2 + image_shape[1] ** 2)
        distance_diff = abs(left_to_neck - right_to_neck) / image_diagonal
        
        # Check if the difference is within symmetry threshold
        return distance_diff < self.HAND_SYMMETRY_THRESHOLD

    def detect_side_pose(self, keypoints, confidences):
        """Detect if the person is in side pose based on shoulder positions"""
        left_shoulder = keypoints[self.KEYPOINTS['left_shoulder']]
        right_shoulder = keypoints[self.KEYPOINTS['right_shoulder']]
        left_shoulder_conf = confidences[self.KEYPOINTS['left_shoulder']]
        right_shoulder_conf = confidences[self.KEYPOINTS['right_shoulder']]
        
        if (left_shoulder_conf < self.CONFIDENCE_THRESHOLD or 
            right_shoulder_conf < self.CONFIDENCE_THRESHOLD):
            return False
            
        # Calculate horizontal distance between shoulders
        shoulder_distance = abs(left_shoulder[0] - right_shoulder[0])
        image_width = max(left_shoulder[0], right_shoulder[0]) * 2  # Estimate image width
        normalized_distance = shoulder_distance / image_width if image_width > 0 else 0
        
        # If shoulders are too close horizontally, it's likely a side pose
        return normalized_distance < self.SIDE_POSE_THRESHOLD

    def calculate_cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        try:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0
                
            return dot_product / (norm1 * norm2)
        except Exception:
            return 0

    def check_arm_direction_to_neck(self, wrist, elbow, neck, confidences, side):
        """Check if arm is directed towards neck using cosine similarity"""
        wrist_conf = confidences[self.KEYPOINTS[f'{side}_wrist']]
        elbow_conf = confidences[self.KEYPOINTS[f'{side}_elbow']]
        
        if (wrist_conf < self.CONFIDENCE_THRESHOLD or 
            elbow_conf < self.CONFIDENCE_THRESHOLD or 
            neck is None):
            return False, 0
            
        # Calculate arm direction vector (wrist - elbow)
        arm_vector = np.array(wrist) - np.array(elbow)
        
        # Calculate neck direction vector (neck - elbow)
        neck_vector = np.array(neck) - np.array(elbow)
        
        # Calculate cosine similarity
        cosine_sim = self.calculate_cosine_similarity(arm_vector, neck_vector)
        
        # Check if arm is pointing towards neck
        is_directed = cosine_sim >= self.ARM_NECK_COSINE_THRESHOLD
        
        return is_directed, cosine_sim

    def check_wrist_elbow_near_neck(self, keypoints, confidences, neck_center, image_shape):
        """Check if wrist or elbow keypoints are near neck area"""
        if neck_center is None:
            return {'left': False, 'right': False, 'any_near': False}
            
        results = {'left': False, 'right': False, 'any_near': False}
        
        for side in ['left', 'right']:
            wrist = keypoints[self.KEYPOINTS[f'{side}_wrist']]
            elbow = keypoints[self.KEYPOINTS[f'{side}_elbow']]
            wrist_conf = confidences[self.KEYPOINTS[f'{side}_wrist']]
            elbow_conf = confidences[self.KEYPOINTS[f'{side}_elbow']]
            
            # Check wrist proximity to neck
            if wrist_conf > self.CONFIDENCE_THRESHOLD:
                wrist_dist = self.calculate_wrist_to_neck_distance(wrist, neck_center, image_shape)
                if wrist_dist < self.WRIST_NECK_THRESHOLD:
                    results[side] = True
                    results['any_near'] = True
            
            # Check elbow proximity to neck
            if elbow_conf > self.CONFIDENCE_THRESHOLD:
                elbow_dist = self.calculate_wrist_to_neck_distance(elbow, neck_center, image_shape)
                if elbow_dist < self.ELBOW_NECK_THRESHOLD:
                    results[side] = True
                    results['any_near'] = True
                    
        return results

    def check_both_hands_at_neck(self, left_wrist, right_wrist, neck_center, image_shape):
        """Check if both hands are positioned close to the neck area"""
        if neck_center is None:
            return False
        
        left_dist = self.calculate_wrist_to_neck_distance(left_wrist, neck_center, image_shape)
        right_dist = self.calculate_wrist_to_neck_distance(right_wrist, neck_center, image_shape)
        diff = abs(left_dist - right_dist)
        
        # Both wrists should be close to neck
        return (left_dist < self.WRIST_NECK_THRESHOLD and 
                right_dist < self.WRIST_NECK_THRESHOLD )
    
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
    
    def detect_violence_like_motion(self, keypoints, confidences):
        """Detect violence-like motions to differentiate from choking"""
        try:
            # Check for punching-like motions (extended arms away from body)
            left_shoulder = keypoints[self.KEYPOINTS['left_shoulder']]
            right_shoulder = keypoints[self.KEYPOINTS['right_shoulder']]
            left_wrist = keypoints[self.KEYPOINTS['left_wrist']]
            right_wrist = keypoints[self.KEYPOINTS['right_wrist']]
            left_elbow = keypoints[self.KEYPOINTS['left_elbow']]
            right_elbow = keypoints[self.KEYPOINTS['right_elbow']]
            
            left_shoulder_conf = confidences[self.KEYPOINTS['left_shoulder']]
            right_shoulder_conf = confidences[self.KEYPOINTS['right_shoulder']]
            left_wrist_conf = confidences[self.KEYPOINTS['left_wrist']]
            right_wrist_conf = confidences[self.KEYPOINTS['right_wrist']]
            left_elbow_conf = confidences[self.KEYPOINTS['left_elbow']]
            right_elbow_conf = confidences[self.KEYPOINTS['right_elbow']]
            
            violence_indicators = 0
            
            # Check for extended arm motions (punching-like)
            if (left_shoulder_conf > self.CONFIDENCE_THRESHOLD and 
                left_wrist_conf > self.CONFIDENCE_THRESHOLD and
                left_elbow_conf > self.CONFIDENCE_THRESHOLD):
                
                # Calculate if left arm is extended outward
                shoulder_to_elbow = np.linalg.norm(np.array(left_elbow) - np.array(left_shoulder))
                elbow_to_wrist = np.linalg.norm(np.array(left_wrist) - np.array(left_elbow))
                shoulder_to_wrist = np.linalg.norm(np.array(left_wrist) - np.array(left_shoulder))
                
                # If arm is relatively straight and extended
                if shoulder_to_wrist > (shoulder_to_elbow + elbow_to_wrist) * 0.8:
                    violence_indicators += 1
            
            if (right_shoulder_conf > self.CONFIDENCE_THRESHOLD and 
                right_wrist_conf > self.CONFIDENCE_THRESHOLD and
                right_elbow_conf > self.CONFIDENCE_THRESHOLD):
                
                # Calculate if right arm is extended outward
                shoulder_to_elbow = np.linalg.norm(np.array(right_elbow) - np.array(right_shoulder))
                elbow_to_wrist = np.linalg.norm(np.array(right_wrist) - np.array(right_elbow))
                shoulder_to_wrist = np.linalg.norm(np.array(right_wrist) - np.array(right_shoulder))
                
                # If arm is relatively straight and extended
                if shoulder_to_wrist > (shoulder_to_elbow + elbow_to_wrist) * 0.8:
                    violence_indicators += 1
            
            # Check for wide arm positioning (fighting stance)
            if (left_wrist_conf > self.CONFIDENCE_THRESHOLD and 
                right_wrist_conf > self.CONFIDENCE_THRESHOLD):
                wrist_distance = np.linalg.norm(np.array(left_wrist) - np.array(right_wrist))
                shoulder_distance = np.linalg.norm(np.array(left_shoulder) - np.array(right_shoulder))
                
                # If wrists are much wider than shoulders (fighting stance)
                if wrist_distance > shoulder_distance * 2.0:
                    violence_indicators += 1
            
            # Return True if multiple violence indicators are present
            return violence_indicators >= 2
            
        except Exception as e:
            logger.error(f"Error detecting violence-like motion: {e}")
            return False

    def analyze_keypoints(self, keypoints, confidences, image_shape):
        """Analyze keypoints for choking detection"""
        features = {
            'both_hands_at_neck': False,
            'hands_symmetric': False,
            'wrists_close': False,
            'individual_wrist_at_neck': False,
            'side_pose_detected': False,
            'arm_directed_to_neck': False,
            'wrist_elbow_near_neck': False,
            'violence_like_motion': False,
            'detection_state': "NORMAL",
            'detection_probability': 0.0,
            'detection_type': 'choking'
        }
        
        neck_center, neck_detected = self.calculate_neck_center(keypoints, confidences)
        if not neck_detected:
            return features

        # Detect side pose
        features['side_pose_detected'] = self.detect_side_pose(keypoints, confidences)
        
        # Check wrist/elbow proximity to neck
        proximity_results = self.check_wrist_elbow_near_neck(keypoints, confidences, neck_center, image_shape)
        features['wrist_elbow_near_neck'] = proximity_results['any_near']
        
        left_wrist = keypoints[self.KEYPOINTS['left_wrist']]
        right_wrist = keypoints[self.KEYPOINTS['right_wrist']]
        left_elbow = keypoints[self.KEYPOINTS['left_elbow']]
        right_elbow = keypoints[self.KEYPOINTS['right_elbow']]
        left_wrist_conf = confidences[self.KEYPOINTS['left_wrist']]
        right_wrist_conf = confidences[self.KEYPOINTS['right_wrist']]
        
        # Check arm direction towards neck using cosine similarity
        left_arm_directed, left_cosine = self.check_arm_direction_to_neck(
            left_wrist, left_elbow, neck_center, confidences, 'left'
        )
        right_arm_directed, right_cosine = self.check_arm_direction_to_neck(
            right_wrist, right_elbow, neck_center, confidences, 'right'
        )
        features['arm_directed_to_neck'] = left_arm_directed or right_arm_directed

        # Both wrists must have sufficient confidence for proper choking detection
        if (left_wrist_conf > self.CONFIDENCE_THRESHOLD and 
            right_wrist_conf > self.CONFIDENCE_THRESHOLD):
            
            # Check if both hands are positioned at neck
            features['both_hands_at_neck'] = self.check_both_hands_at_neck(
                left_wrist, right_wrist, neck_center, image_shape
            )
            
            # Check hand symmetry (both hands equidistant from neck)
            features['hands_symmetric'] = self.check_hand_symmetry(
                left_wrist, right_wrist, neck_center, image_shape
            )
            
            # Check if wrists are close to each other
            wrist_distance = self.calculate_wrist_distance(left_wrist, right_wrist, image_shape)
            features['wrists_close'] = wrist_distance < self.WRIST_DISTANCE_THRESHOLD
            
        else:
            # Check individual wrists if only one has good confidence
            left_at_neck = False
            right_at_neck = False
            
            if left_wrist_conf > self.CONFIDENCE_THRESHOLD:
                left_dist = self.calculate_wrist_to_neck_distance(left_wrist, neck_center, image_shape)
                left_at_neck = left_dist < self.WRIST_NECK_THRESHOLD
                
            if right_wrist_conf > self.CONFIDENCE_THRESHOLD:
                right_dist = self.calculate_wrist_to_neck_distance(right_wrist, neck_center, image_shape)
                right_at_neck = right_dist < self.WRIST_NECK_THRESHOLD
                
            features['individual_wrist_at_neck'] = left_at_neck or right_at_neck

        # Detect potential violence-like motions to reduce false positives
        features['violence_like_motion'] = self.detect_violence_like_motion(keypoints, confidences)
        
        # Enhanced choking detection logic with violence differentiation
        if features['violence_like_motion']:
            # Reduce probability if violence-like motion detected
            features['detection_state'] = "POSSIBLE VIOLENCE (NOT CHOKING)"
            features['detection_probability'] = 0.1
        elif (features['both_hands_at_neck'] and 
              features['hands_symmetric'] and 
              features['wrists_close'] and
              features['arm_directed_to_neck']):
            features['detection_state'] = "CHOKING (HIGH CONFIDENCE)"
            features['detection_probability'] = 0.95
        elif (features['side_pose_detected'] and 
              features['wrist_elbow_near_neck'] and 
              features['arm_directed_to_neck']):
            features['detection_state'] = "CHOKING (SIDE POSE)"
            features['detection_probability'] = 0.9
        elif features['both_hands_at_neck'] and features['wrists_close']:
            features['detection_state'] = "CHOKING (BOTH HANDS CLOSE)"
            features['detection_probability'] = 0.85
        elif features['both_hands_at_neck'] and features['arm_directed_to_neck']:
            features['detection_state'] = "CHOKING (DIRECTED ARMS)"
            features['detection_probability'] = 0.8
        elif features['both_hands_at_neck']:
            features['detection_state'] = "CHOKING (BOTH HANDS AT NECK)"
            features['detection_probability'] = 0.75
        elif features['wrist_elbow_near_neck'] and features['arm_directed_to_neck']:
            features['detection_state'] = "POSSIBLE CHOKING (ARM DIRECTED)"
            features['detection_probability'] = 0.7
        elif features['individual_wrist_at_neck']:
            features['detection_state'] = "POSSIBLE CHOKING (ONE HAND)"
            features['detection_probability'] = 0.6
        else:
            features['detection_state'] = "NORMAL"
            features['detection_probability'] = 0.0

        return features

    def draw_annotations(self, frame, keypoints, confidences, features, confidence_threshold):
        """Draw choking-specific annotations on frame"""
        try:
            # Draw skeleton and keypoints
            frame = self.draw_skeleton(frame, keypoints, confidences, color=(0, 255, 0), thickness=2)
            frame = self.draw_keypoints(frame, keypoints, confidences, color=(0, 255, 0), radius=5)
            
            neck_center, neck_detected = self.calculate_neck_center(keypoints, confidences)
            left_wrist = keypoints[self.KEYPOINTS['left_wrist']]
            right_wrist = keypoints[self.KEYPOINTS['right_wrist']]
            left_wrist_conf = confidences[self.KEYPOINTS['left_wrist']]
            right_wrist_conf = confidences[self.KEYPOINTS['right_wrist']]
            
            if neck_detected:
                # Draw neck center
                cv2.circle(frame, (int(neck_center[0]), int(neck_center[1])), 8, (255, 0, 255), -1)
                
                # Draw wrist-to-neck connections when both hands are at neck
                if (features['both_hands_at_neck'] and 
                    left_wrist_conf > self.CONFIDENCE_THRESHOLD and 
                    right_wrist_conf > self.CONFIDENCE_THRESHOLD):
                    
                    # Red lines to show both hands at neck
                    cv2.line(frame, (int(left_wrist[0]), int(left_wrist[1])), 
                            (int(neck_center[0]), int(neck_center[1])), (0, 0, 255), 3)
                    cv2.line(frame, (int(right_wrist[0]), int(right_wrist[1])), 
                            (int(neck_center[0]), int(neck_center[1])), (0, 0, 255), 3)
                    
                    # Draw line between wrists to show they're close
                    if features['wrists_close']:
                        cv2.line(frame, (int(left_wrist[0]), int(left_wrist[1])), 
                                (int(right_wrist[0]), int(right_wrist[1])), (255, 0, 255), 2)
                    
                    # Highlight symmetry with circles
                    if features['hands_symmetric']:
                        cv2.circle(frame, (int(left_wrist[0]), int(left_wrist[1])), 12, (0, 255, 255), 2)
                        cv2.circle(frame, (int(right_wrist[0]), int(right_wrist[1])), 12, (0, 255, 255), 2)
                
                # Draw individual wrist connections for partial detection
                elif features['individual_wrist_at_neck']:
                    if (left_wrist_conf > self.CONFIDENCE_THRESHOLD and 
                        self.calculate_wrist_to_neck_distance(left_wrist, neck_center, frame.shape[:2]) < self.WRIST_NECK_THRESHOLD):
                        cv2.line(frame, (int(left_wrist[0]), int(left_wrist[1])), 
                                (int(neck_center[0]), int(neck_center[1])), (0, 165, 255), 2)
                    
                    if (right_wrist_conf > self.CONFIDENCE_THRESHOLD and 
                        self.calculate_wrist_to_neck_distance(right_wrist, neck_center, frame.shape[:2]) < self.WRIST_NECK_THRESHOLD):
                        cv2.line(frame, (int(right_wrist[0]), int(right_wrist[1])), 
                                (int(neck_center[0]), int(neck_center[1])), (0, 165, 255), 2)
                
        except Exception as e:
            logger.error(f"Error drawing choking annotations: {e}")
        
        return frame

    def enhance_with_mediapipe(self, frame, yolo_features):
        """Enhance detection using MediaPipe for additional precision"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_results = self.pose.process(rgb_frame)
            
            if mp_results.pose_landmarks:
                # Extract MediaPipe landmarks
                landmarks = mp_results.pose_landmarks.landmark
                
                # Get key points from MediaPipe
                mp_left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
                mp_right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
                mp_left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW]
                mp_right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
                
                # Convert to pixel coordinates
                h, w = frame.shape[:2]
                mp_left_wrist_px = (int(mp_left_wrist.x * w), int(mp_left_wrist.y * h))
                mp_right_wrist_px = (int(mp_right_wrist.x * w), int(mp_right_wrist.y * h))
                
                # Check if MediaPipe detects hands near neck area
                neck_y = int((landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y + 
                             landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y) * h / 2)
                
                # Enhanced detection if MediaPipe confirms hand positions
                if (abs(mp_left_wrist_px[1] - neck_y) < h * 0.1 or 
                    abs(mp_right_wrist_px[1] - neck_y) < h * 0.1):
                    yolo_features['detection_probability'] = min(1.0, yolo_features['detection_probability'] * 1.2)
                    
            return yolo_features
            
        except Exception as e:
            logger.error(f"Error in MediaPipe enhancement: {e}")
            return yolo_features
    
    def analyze(self, frame, confidence, alert_threshold, finger_tips=None):
        try:
            results = self.model(frame, conf=confidence, verbose=False)
            annotated_frame = frame.copy()
            max_probability = 0.0
            detected = False
            best_features = None
            
            for result in results:
                if result.keypoints is not None:
                    keypoints = self.safe_tensor_to_numpy(result.keypoints.xy)
                    confidences = self.safe_tensor_to_numpy(result.keypoints.conf)
                    
                    if keypoints.size == 0 or confidences.size == 0:
                        continue
                    
                    for kpts, confs in zip(keypoints, confidences):
                        features = self.analyze_keypoints(kpts, confs, frame.shape[:2])
                        
                        # Enhance with MediaPipe if probability is significant
                        if features['detection_probability'] > 0.5:
                            features = self.enhance_with_mediapipe(frame, features)
                        
                        if features['detection_probability'] > max_probability:
                            max_probability = features['detection_probability']
                            best_features = features
                        
                        annotated_frame = self.draw_annotations(
                            annotated_frame, kpts, confs, features, confidence
                        )
                        
                        if features['detection_probability'] >= alert_threshold:
                            detected = True
            
            detection_status = {
                'type': 'choking',
                'detected': detected,
                'probability': max_probability,
                'state': best_features['detection_state'] if best_features else 'NORMAL',
                'features': best_features if best_features else {}
            }
            
            return annotated_frame, detection_status
            
        except Exception as e:
            logger.error(f"Error in choking analysis: {e}")
            detection_status = {
                'type': 'choking',
                'detected': False,
                'probability': 0.0,
                'state': 'ERROR'
            }
            return frame, detection_status