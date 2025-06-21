import cv2
import numpy as np
import math
from ultralytics import YOLO
import mediapipe as mp
import logging
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChokingDetector:
    """
    Choking detection class that processes frames and returns annotated frames with choking detection.
    
    Usage:
        detector = ChokingDetector()
        annotated_frame = detector.detect(frame, confidence_threshold=0.5, alert_threshold=0.7)
    """
    
    def __init__(self, model_path='yolov8m-pose.pt'):
        """
        Initialize the choking detector.
        
        Args:
            model_path (str): Path to the YOLO pose model
        """
        # Check GPU availability
        self.device = self._check_gpu()
        logger.info(f"Using device: {self.device}")
        
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.analyzer = ChokingAnalyzer()
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
    
    def _check_gpu(self):
        """Check GPU availability and return appropriate device."""
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            logger.info(f"CUDA is available with {device_count} GPU(s)")
            return 'cuda'
        else:
            logger.info("CUDA is not available, using CPU")
            return 'cpu'
    
    def detect(self, frame, confidence_threshold=0.5, alert_threshold=0.7):
        """
        Process a frame and return annotated frame with choking detection.
        
        Args:
            frame (numpy.ndarray): Input frame (BGR format)
            confidence_threshold (float): Confidence threshold for pose detection (0.1-1.0)
            alert_threshold (float): Alert threshold for choking detection (0.1-1.0)
        
        Returns:
            tuple: (annotated_frame, choking_probability)
                - annotated_frame (numpy.ndarray): Annotated frame with detection results
                - choking_probability (float): Probability of choking (0.0-1.0)
        """
        try:
            # Run YOLO pose detection
            results = self.model(frame, conf=confidence_threshold, verbose=False)
            annotated_frame = frame.copy()
            
            choking_detected = False
            max_probability = 0.0
            
            for result in results:
                if result.keypoints is not None:
                    # Handle device-specific tensor operations
                    if self.device == 'cuda':
                        keypoints = result.keypoints.xy.cpu().numpy()
                        confidences = result.keypoints.conf.cpu().numpy()
                    else:
                        keypoints = result.keypoints.xy.numpy()
                        confidences = result.keypoints.conf.numpy()
                    
                    for kpts, confs in zip(keypoints, confidences):
                        # Detect finger tips using MediaPipe
                        finger_tips = self._detect_fingers(frame, kpts, confs)
                        
                        # Analyze choking features
                        features = self.analyzer.analyze(
                            kpts, confs, frame.shape[:2], finger_tips=finger_tips
                        )
                        
                        max_probability = max(max_probability, features['choking_probability'])
                        
                        # Draw annotations on frame
                        annotated_frame = self._draw_annotations(
                            annotated_frame, kpts, confs, features, confidence_threshold
                        )
                        
                        # Check if choking is detected based on the specific condition
                        if features['choking_state'] == "CHOKING":
                            choking_detected = True
            
            return annotated_frame, max_probability
            
        except Exception as e:
            logger.error(f"Error in choking detection: {e}")
            return frame, 0.0
    
    def _detect_fingers(self, frame, keypoints, confidences):
        """Detect fingers using MediaPipe Hands, returns list of finger tip positions."""
        finger_tips = []
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Finger tip landmark indices: Thumb_tip, Index_tip, Middle_tip, Ring_tip, Pinky_tip
                for idx in [4, 8, 12, 16, 20]:
                    lm = hand_landmarks.landmark[idx]
                    h, w, _ = frame.shape
                    x, y = int(lm.x * w), int(lm.y * h)
                    finger_tips.append((x, y))
        
        return finger_tips
    
    def _draw_annotations(self, frame, keypoints, confidences, features, confidence_threshold):
        """Draw annotations on the frame."""
        # Draw keypoints
        for i, (x, y) in enumerate(keypoints):
            if confidences[i] > confidence_threshold:
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
        
        # Draw neck center
        neck_center, neck_detected = self.analyzer.calculate_neck_center(keypoints, confidences)
        if neck_detected:
            cv2.circle(frame, (int(neck_center[0]), int(neck_center[1])), 8, (255, 0, 255), -1)
        
        # Draw wrist-to-neck lines
        if features['left_wrist_at_neck'] and neck_detected:
            lw = keypoints[self.analyzer.KEYPOINTS['left_wrist']]
            cv2.line(frame, (int(lw[0]), int(lw[1])), 
                    (int(neck_center[0]), int(neck_center[1])), (0, 0, 255), 2)
        
        if features['right_wrist_at_neck'] and neck_detected:
            rw = keypoints[self.analyzer.KEYPOINTS['right_wrist']]
            cv2.line(frame, (int(rw[0]), int(rw[1])), 
                    (int(neck_center[0]), int(neck_center[1])), (0, 0, 255), 2)
        
        # Draw eyes closed indicator
        if features['eyes_closed']:
            for eye in ['left_eye', 'right_eye']:
                idx = self.analyzer.KEYPOINTS[eye]
                if confidences[idx] > confidence_threshold:
                    x, y = int(keypoints[idx][0]), int(keypoints[idx][1])
                    cv2.line(frame, (x-8, y-8), (x+8, y+8), (0, 100, 255), 3)
                    cv2.line(frame, (x-8, y+8), (x+8, y-8), (0, 100, 255), 3)
        
        # Draw finger tips
        finger_tips = self._detect_fingers(frame, keypoints, confidences)
        for x, y in finger_tips:
            cv2.circle(frame, (x, y), 5, (255, 255, 0), -1)
        
        # Draw status text
        color = (0, 0, 255) if features['choking_probability'] > 0.7 else (0, 255, 0)
        cv2.putText(frame, features['choking_state'], (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Draw probability
        cv2.putText(frame, f"Probability: {features['choking_probability']:.2f}", 
                   (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return frame


class ChokingAnalyzer:
    """Analyzes choking signs based on pose keypoints and features"""

    def __init__(self):
        self.KEYPOINTS = {
            'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
            'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
            'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
            'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
        }
        self.WRIST_NECK_THRESHOLD = 0.12
        self.CONFIDENCE_THRESHOLD = 0.3
        self.eye_closed_threshold = 15  # pixels

    def calculate_neck_center(self, keypoints, confidences):
        """Calculate the center point of the neck based on shoulder keypoints."""
        left_shoulder = keypoints[self.KEYPOINTS['left_shoulder']]
        right_shoulder = keypoints[self.KEYPOINTS['right_shoulder']]
        left_shoulder_conf = confidences[self.KEYPOINTS['left_shoulder']]
        right_shoulder_conf = confidences[self.KEYPOINTS['right_shoulder']]
        
        if left_shoulder_conf > self.CONFIDENCE_THRESHOLD and right_shoulder_conf > self.CONFIDENCE_THRESHOLD:
            neck_x = (left_shoulder[0] + right_shoulder[0]) / 2
            neck_y = (left_shoulder[1] + right_shoulder[1]) / 2 + 10  # offset down
            return np.array([neck_x, neck_y]), True
        return None, False

    def calculate_wrist_to_neck_distance(self, wrist_point, neck_center, image_shape):
        """Calculate normalized distance between wrist and neck."""
        if neck_center is None:
            return float('inf')
        distance = np.linalg.norm(np.array(wrist_point) - np.array(neck_center))
        image_diagonal = math.sqrt(image_shape[0] ** 2 + image_shape[1] ** 2)
        return distance / image_diagonal

    def detect_eye_closed(self, keypoints, confidences):
        """Detect if eyes are closed based on eye-ear distance."""
        left_eye = keypoints[self.KEYPOINTS['left_eye']]
        right_eye = keypoints[self.KEYPOINTS['right_eye']]
        left_ear = keypoints[self.KEYPOINTS['left_ear']]
        right_ear = keypoints[self.KEYPOINTS['right_ear']]
        
        left_eye_conf = confidences[self.KEYPOINTS['left_eye']]
        right_eye_conf = confidences[self.KEYPOINTS['right_eye']]
        left_ear_conf = confidences[self.KEYPOINTS['left_ear']]
        right_ear_conf = confidences[self.KEYPOINTS['right_ear']]
        
        if (left_eye_conf > self.CONFIDENCE_THRESHOLD and right_eye_conf > self.CONFIDENCE_THRESHOLD and
            left_ear_conf > self.CONFIDENCE_THRESHOLD and right_ear_conf > self.CONFIDENCE_THRESHOLD):
            left_eye_ear_dist = np.linalg.norm(np.array(left_eye) - np.array(left_ear))
            right_eye_ear_dist = np.linalg.norm(np.array(right_eye) - np.array(right_ear))
            
            if left_eye_ear_dist < self.eye_closed_threshold or right_eye_ear_dist < self.eye_closed_threshold:
                return True
        return False

    def analyze(self, keypoints, confidences, image_shape, finger_tips=None):
        """
        Analyze keypoints and finger tips to determine choking features.
        
        Returns:
            dict: Features dictionary with choking analysis results
        """
        features = {
            'left_wrist_at_neck': False,
            'right_wrist_at_neck': False,
            'both_wrists_at_neck': False,
            'finger_at_neck': False,
            'eyes_closed': False,
            'choking_state': "NORMAL",
            'choking_probability': 0.0
        }
        
        # Calculate neck center
        neck_center, neck_detected = self.calculate_neck_center(keypoints, confidences)
        if not neck_detected:
            return features

        # Check wrist positions
        left_wrist = keypoints[self.KEYPOINTS['left_wrist']]
        right_wrist = keypoints[self.KEYPOINTS['right_wrist']]
        left_wrist_conf = confidences[self.KEYPOINTS['left_wrist']]
        right_wrist_conf = confidences[self.KEYPOINTS['right_wrist']]

        if left_wrist_conf > self.CONFIDENCE_THRESHOLD:
            left_dist = self.calculate_wrist_to_neck_distance(left_wrist, neck_center, image_shape)
            if left_dist < self.WRIST_NECK_THRESHOLD:
                features['left_wrist_at_neck'] = True

        if right_wrist_conf > self.CONFIDENCE_THRESHOLD:
            right_dist = self.calculate_wrist_to_neck_distance(right_wrist, neck_center, image_shape)
            if right_dist < self.WRIST_NECK_THRESHOLD:
                features['right_wrist_at_neck'] = True

        features['both_wrists_at_neck'] = features['left_wrist_at_neck'] and features['right_wrist_at_neck']
        features['eyes_closed'] = self.detect_eye_closed(keypoints, confidences)

        # Check finger-to-neck proximity
        FINGER_NECK_THRESHOLD = 0.10
        if finger_tips:
            for fx, fy in finger_tips:
                finger_dist = np.linalg.norm(np.array([fx, fy]) - np.array(neck_center))
                image_diagonal = math.sqrt(image_shape[0] ** 2 + image_shape[1] ** 2)
                norm_finger_dist = finger_dist / image_diagonal
                if norm_finger_dist < FINGER_NECK_THRESHOLD:
                    features['finger_at_neck'] = True
                    break

        # Determine choking state
        if features['eyes_closed']:
            features['choking_state'] = "CHOKING (EYES CLOSED)"
            features['choking_probability'] = 1.0
        elif features['both_wrists_at_neck'] or features['finger_at_neck']:
            features['choking_state'] = "CHOKING"
            features['choking_probability'] = 0.9
        elif features['left_wrist_at_neck'] or features['right_wrist_at_neck']:
            features['choking_state'] = "CHOKING (ONE WRIST)"
            features['choking_probability'] = 0.7
        else:
            features['choking_state'] = "NORMAL"
            features['choking_probability'] = 0.0

        return features


# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = ChokingDetector()
    
    # Example with webcam
    cap = cv2.VideoCapture("rtsp://admin:@2.55.92.197/play1.sdp")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame with custom parameters
        annotated_frame, probability = detector.detect(
            frame, 
            confidence_threshold=0.5, 
            alert_threshold=0.7
        )
        cv2.imshow("Choking Detector", annotated_frame)
        if probability == 0.9:
            cv2.imwrite("choking.jpg", annotated_frame)
        
        # Display result
        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()