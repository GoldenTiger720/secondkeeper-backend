import numpy as np
import math
import cv2


class BaseDetector:
    """Base analyzer class with common functionality"""
    
    def __init__(self):
        self.KEYPOINTS = {
            'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
            'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
            'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
            'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
        }
        self.CONFIDENCE_THRESHOLD = 0.3
        
        # Define skeleton connections (COCO format)
        self.SKELETON_CONNECTIONS = [
            # Head connections
            ('left_eye', 'nose'), ('right_eye', 'nose'),
            ('left_eye', 'left_ear'), ('right_eye', 'right_ear'),
            
            # Upper body connections
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
            
            # Torso connections
            ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
            
            # Lower body connections
            ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
            ('right_hip', 'right_knee'), ('right_knee', 'right_ankle')
        ]

    def calculate_neck_center(self, keypoints, confidences):
        """Calculate neck center based on shoulder positions"""
        left_shoulder = keypoints[self.KEYPOINTS['left_shoulder']]
        right_shoulder = keypoints[self.KEYPOINTS['right_shoulder']]
        left_shoulder_conf = confidences[self.KEYPOINTS['left_shoulder']]
        right_shoulder_conf = confidences[self.KEYPOINTS['right_shoulder']]
        
        if left_shoulder_conf > self.CONFIDENCE_THRESHOLD and right_shoulder_conf > self.CONFIDENCE_THRESHOLD:
            neck_x = (left_shoulder[0] + right_shoulder[0]) / 2
            neck_y = (left_shoulder[1] + right_shoulder[1]) / 2
            return np.array([neck_x, neck_y]), True
        return None, False

    def calculate_body_center(self, keypoints, confidences):
        """Calculate center point of the body (hip area)"""
        left_hip = keypoints[self.KEYPOINTS['left_hip']]
        right_hip = keypoints[self.KEYPOINTS['right_hip']]
        left_hip_conf = confidences[self.KEYPOINTS['left_hip']]
        right_hip_conf = confidences[self.KEYPOINTS['right_hip']]
        
        if left_hip_conf > self.CONFIDENCE_THRESHOLD and right_hip_conf > self.CONFIDENCE_THRESHOLD:
            center_x = (left_hip[0] + right_hip[0]) / 2
            center_y = (left_hip[1] + right_hip[1]) / 2
            return np.array([center_x, center_y]), True
        return None, False

    def draw_skeleton(self, frame, keypoints, confidences, color=(0, 255, 0), thickness=2):
        """Draw skeleton connections between keypoints"""
        for connection in self.SKELETON_CONNECTIONS:
            point1_name, point2_name = connection
            point1_idx = self.KEYPOINTS[point1_name]
            point2_idx = self.KEYPOINTS[point2_name]
            
            # Check if both points have sufficient confidence
            if (confidences[point1_idx] > self.CONFIDENCE_THRESHOLD and 
                confidences[point2_idx] > self.CONFIDENCE_THRESHOLD):
                point1 = keypoints[point1_idx]
                point2 = keypoints[point2_idx]
                if point1[0] > 0 and point1[1] > 0 and point2[0] > 0 and point2[1] > 0:
                # Draw line between the two points
                    cv2.line(frame, 
                            (int(point1[0]), int(point1[1])), 
                            (int(point2[0]), int(point2[1])), 
                            color, thickness)
            
        return frame

    def draw_keypoints(self, frame, keypoints, confidences, color=(0, 255, 0), radius=2):
        """Draw individual keypoints as circles"""
        for i, (keypoint, conf) in enumerate(zip(keypoints, confidences)):
            if conf > self.CONFIDENCE_THRESHOLD and int(keypoint[0]) > 0 and int(keypoint[1]) > 0 :
                cv2.circle(frame, (int(keypoint[0]), int(keypoint[1])), radius, color, -1)
        return frame

    def analyze(self, frame, confidence, alert_threshold, finger_tips=None):
        raise NotImplementedError("Subclasses must implement analyze method")