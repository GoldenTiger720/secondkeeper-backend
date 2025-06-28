import cv2
import numpy as np
import math
import logging
from ultralytics import YOLO
from .base_detector import BaseDetector
from collections import defaultdict, deque
from django.conf import settings

logger = logging.getLogger("security_ai")

class ViolenceTracker:
    """Tracks person behavior over time for temporal analysis"""
    
    def __init__(self, history_length=30):  # 1 second at 30fps
        self.history = defaultdict(lambda: deque(maxlen=history_length))
        self.history_length = history_length
    
    def update_person_history(self, person_id, features):
        """Update individual person's behavior history"""
        self.history[person_id].append(features)
    
    def analyze_movement_pattern(self, person_id):
        """Analyze movement patterns for rapid/aggressive movements"""
        if person_id not in self.history or len(self.history[person_id]) < 10:
            return {'pattern': 'insufficient_data', 'risk_score': 0.0}
        
        recent_positions = [h.get('center') for h in self.history[person_id] if h.get('center') is not None]
        
        if len(recent_positions) < 5:
            return {'pattern': 'insufficient_data', 'risk_score': 0.0}
        
        # Calculate velocities
        velocities = []
        for i in range(1, len(recent_positions)):
            velocity = np.linalg.norm(recent_positions[i] - recent_positions[i-1])
            velocities.append(velocity)
        
        if velocities:
            avg_velocity = np.mean(velocities)
            max_velocity = np.max(velocities)
            velocity_variance = np.var(velocities)
            
            # Determine movement pattern
            risk_score = 0.0
            pattern = 'normal'
            
            # Rapid movement detection
            if max_velocity > avg_velocity * 3:
                pattern = 'rapid_movement'
                risk_score += 0.3
            
            # Erratic movement detection
            if velocity_variance > avg_velocity * 2:
                pattern = 'erratic_movement'
                risk_score += 0.2
            
            # Sustained high velocity
            high_velocity_count = sum(1 for v in velocities[-10:] if v > avg_velocity * 1.5)
            if high_velocity_count > 5:
                pattern = 'sustained_aggression'
                risk_score += 0.4
            
            return {
                'pattern': pattern,
                'risk_score': min(risk_score, 1.0),
                'avg_velocity': avg_velocity,
                'max_velocity': max_velocity
            }
        
        return {'pattern': 'static', 'risk_score': 0.0}
    
    def get_pose_consistency(self, person_id):
        """Analyze consistency of aggressive poses over time"""
        if person_id not in self.history or len(self.history[person_id]) < 15:
            return 0.0
        
        recent_poses = list(self.history[person_id])[-15:]
        aggressive_count = sum(1 for pose in recent_poses 
                             if pose.get('aggressive_features', {}).get('total_aggression_score', 0) > 0.5)
        
        return aggressive_count / len(recent_poses)


class ViolenceDetector(BaseDetector):
    """Enhanced violence analyzer with multi-person interaction analysis"""

    def __init__(self):
        super().__init__()
        self.raised_arm_threshold = 0.2
        self.aggressive_distance_threshold = 0.25
        self.minimum_people_for_violence = 2
        model_path=settings.MODEL_PATHS['pose_model']
        self.model = YOLO(model_path)
        self.tracker = ViolenceTracker()
        
        # Interaction zones (normalized distances)
        self.interaction_zones = {
            'intimate': 0.15,      # Very close - high violence potential
            'personal': 0.25,      # Personal space invasion
            'social': 0.4,         # Social distance
            'public': float('inf') # Public distance
        }

    def detect_multiple_persons(self, results):
        """Detect and extract information for multiple persons"""
        persons = []
        
        for result in results:
            if result.keypoints is not None and result.boxes is not None:
                keypoints = self.safe_tensor_to_numpy(result.keypoints.xy)
                confidences = self.safe_tensor_to_numpy(result.keypoints.conf)
                boxes = self.safe_tensor_to_numpy(result.boxes.xyxy)
                
                if keypoints.size == 0 or confidences.size == 0:
                    continue
                
                for i, (kpts, confs, box) in enumerate(zip(keypoints, confidences, boxes)):
                    # Validate person detection
                    key_points_detected = sum([
                        confs[self.KEYPOINTS['nose']] > self.CONFIDENCE_THRESHOLD,
                        confs[self.KEYPOINTS['left_shoulder']] > self.CONFIDENCE_THRESHOLD,
                        confs[self.KEYPOINTS['right_shoulder']] > self.CONFIDENCE_THRESHOLD,
                        confs[self.KEYPOINTS['left_hip']] > self.CONFIDENCE_THRESHOLD,
                        confs[self.KEYPOINTS['right_hip']] > self.CONFIDENCE_THRESHOLD
                    ])
                    
                    if key_points_detected >= 2:
                        center = self.get_person_center(kpts, confs)
                        if center is not None:
                            person = {
                                'id': len(persons),
                                'keypoints': kpts,
                                'confidences': confs,
                                'bbox': box,
                                'center': center,
                                'body_size': self.estimate_body_size(kpts, confs)
                            }
                            persons.append(person)
        
        return persons

    def get_person_center(self, keypoints, confidences):
        """Calculate person's center point from valid keypoints"""
        # Use torso keypoints for center calculation
        torso_points = []
        torso_keypoints = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
        
        for kp_name in torso_keypoints:
            if confidences[self.KEYPOINTS[kp_name]] > self.CONFIDENCE_THRESHOLD:
                torso_points.append(keypoints[self.KEYPOINTS[kp_name]])
        
        if len(torso_points) >= 2:
            return np.mean(torso_points, axis=0)
        return None

    def estimate_body_size(self, keypoints, confidences):
        """Estimate person's body size for normalization"""
        # Calculate distance between shoulders and hips
        if (confidences[self.KEYPOINTS['left_shoulder']] > self.CONFIDENCE_THRESHOLD and
            confidences[self.KEYPOINTS['left_hip']] > self.CONFIDENCE_THRESHOLD):
            shoulder = keypoints[self.KEYPOINTS['left_shoulder']]
            hip = keypoints[self.KEYPOINTS['left_hip']]
            return np.linalg.norm(shoulder - hip)
        elif (confidences[self.KEYPOINTS['right_shoulder']] > self.CONFIDENCE_THRESHOLD and
              confidences[self.KEYPOINTS['right_hip']] > self.CONFIDENCE_THRESHOLD):
            shoulder = keypoints[self.KEYPOINTS['right_shoulder']]
            hip = keypoints[self.KEYPOINTS['right_hip']]
            return np.linalg.norm(shoulder - hip)
        return 100  # Default body size

    def analyze_spatial_relationship(self, person1, person2, frame_shape):
        """Analyze spatial relationship between two persons"""
        center1 = person1['center']
        center2 = person2['center']
        
        if center1 is None or center2 is None:
            return None
        
        # Calculate distance
        distance = np.linalg.norm(center1 - center2)
        
        # Normalize distance by frame diagonal and average body size
        frame_diagonal = np.sqrt(frame_shape[0]**2 + frame_shape[1]**2)
        avg_body_size = (person1['body_size'] + person2['body_size']) / 2
        normalized_distance = distance / (frame_diagonal * 0.5)
        body_relative_distance = distance / avg_body_size
        
        # Determine interaction zone
        zone = 'public'
        for zone_name, threshold in self.interaction_zones.items():
            if normalized_distance < threshold:
                zone = zone_name
                break
        
        # Check if they're facing each other
        facing_each_other = self.are_facing_each_other(person1, person2)
        
        # Calculate approach direction and speed
        relative_position = center2 - center1
        
        return {
            'distance': distance,
            'normalized_distance': normalized_distance,
            'body_relative_distance': body_relative_distance,
            'zone': zone,
            'facing_each_other': facing_each_other,
            'relative_position': relative_position,
            'interaction_risk': self.calculate_interaction_risk(zone, facing_each_other, body_relative_distance)
        }

    def are_facing_each_other(self, person1, person2):
        """Determine if two persons are facing each other"""
        def get_body_orientation(person):
            kpts = person['keypoints']
            confs = person['confidences']
            
            left_shoulder = kpts[self.KEYPOINTS['left_shoulder']]
            right_shoulder = kpts[self.KEYPOINTS['right_shoulder']]
            
            if (confs[self.KEYPOINTS['left_shoulder']] > 0.5 and
                confs[self.KEYPOINTS['right_shoulder']] > 0.5):
                
                shoulder_vector = right_shoulder - left_shoulder
                # Body direction is perpendicular to shoulder line
                body_direction = np.array([-shoulder_vector[1], shoulder_vector[0]])
                if np.linalg.norm(body_direction) > 0:
                    return body_direction / np.linalg.norm(body_direction)
            return None
        
        orientation1 = get_body_orientation(person1)
        orientation2 = get_body_orientation(person2)
        
        if orientation1 is not None and orientation2 is not None:
            # Vector between persons
            person_vector = person2['center'] - person1['center']
            person_vector = person_vector / np.linalg.norm(person_vector)
            
            # Check if each person is oriented toward the other
            facing_score1 = np.dot(orientation1, person_vector)
            facing_score2 = np.dot(orientation2, -person_vector)
            
            return facing_score1 > 0.3 and facing_score2 > 0.3
        
        return False

    def calculate_interaction_risk(self, zone, facing_each_other, body_relative_distance):
        """Calculate risk score based on spatial interaction"""
        risk_score = 0.0
        
        # Zone-based risk
        zone_risks = {
            'intimate': 0.6,
            'personal': 0.3,
            'social': 0.1,
            'public': 0.0
        }
        risk_score += zone_risks.get(zone, 0.0)
        
        # Facing each other increases risk
        if facing_each_other:
            risk_score += 0.2
        
        # Very close proximity (body-relative)
        if body_relative_distance < 1.5:
            risk_score += 0.3
        
        return min(risk_score, 1.0)

    def analyze_aggressive_pose_advanced(self, person, opponent_position=None):
        """Advanced aggressive pose analysis"""
        keypoints = person['keypoints']
        confidences = person['confidences']
        
        features = {
            'raised_arms': 0,
            'forward_lean': False,
            'wide_stance': False,
            'pointing_gesture': False,
            'clenched_posture': False,
            'total_aggression_score': 0.0
        }
        
        # Enhanced raised arms detection
        features['raised_arms'] = self.detect_raised_arms_advanced(keypoints, confidences)
        
        # Forward lean toward opponent
        if opponent_position is not None:
            features['forward_lean'] = self.detect_forward_lean(keypoints, confidences, opponent_position)
        
        # Wide stance detection
        features['wide_stance'] = self.detect_wide_stance_advanced(keypoints, confidences)
        
        # Pointing/threatening gesture
        if opponent_position is not None:
            features['pointing_gesture'] = self.detect_pointing_gesture(keypoints, confidences, opponent_position)
        
        # Clenched/tense posture
        features['clenched_posture'] = self.detect_clenched_posture(keypoints, confidences)
        
        # Calculate total aggression score
        score = 0.0
        score += features['raised_arms'] * 0.15  # 0-0.3
        score += 0.2 if features['forward_lean'] else 0.0
        score += 0.15 if features['wide_stance'] else 0.0
        score += 0.3 if features['pointing_gesture'] else 0.0
        score += 0.1 if features['clenched_posture'] else 0.0
        
        features['total_aggression_score'] = min(score, 1.0)
        
        return features

    def detect_raised_arms_advanced(self, keypoints, confidences):
        """Enhanced raised arms detection with angle analysis"""
        raised_count = 0
        
        for side in ['left', 'right']:
            shoulder = keypoints[self.KEYPOINTS[f'{side}_shoulder']]
            elbow = keypoints[self.KEYPOINTS[f'{side}_elbow']]
            wrist = keypoints[self.KEYPOINTS[f'{side}_wrist']]
            
            if (confidences[self.KEYPOINTS[f'{side}_shoulder']] > self.CONFIDENCE_THRESHOLD and
                confidences[self.KEYPOINTS[f'{side}_elbow']] > self.CONFIDENCE_THRESHOLD and
                confidences[self.KEYPOINTS[f'{side}_wrist']] > self.CONFIDENCE_THRESHOLD):
                
                # Check if arm is raised above shoulder level
                if elbow[1] < shoulder[1] - 20:  # Elbow above shoulder
                    raised_count += 0.5
                
                # Check if hand is raised significantly
                if wrist[1] < shoulder[1] - 30:  # Wrist well above shoulder
                    raised_count += 0.5
                
                # Check arm extension (aggressive vs defensive)
                upper_arm = elbow - shoulder
                forearm = wrist - elbow
                if np.linalg.norm(upper_arm) > 0 and np.linalg.norm(forearm) > 0:
                    # Calculate angle between upper arm and forearm
                    dot_product = np.dot(upper_arm, forearm)
                    norms = np.linalg.norm(upper_arm) * np.linalg.norm(forearm)
                    if norms > 0:
                        angle = np.arccos(np.clip(dot_product / norms, -1.0, 1.0))
                        # Extended arm (straighter) is more aggressive
                        if angle > np.pi * 0.7:  # More than 126 degrees
                            raised_count += 0.3
        
        return min(raised_count, 2.0)

    def detect_forward_lean(self, keypoints, confidences, opponent_pos):
        """Detect forward lean toward opponent"""
        nose = keypoints[self.KEYPOINTS['nose']]
        hip_center = self.get_hip_center(keypoints, confidences)
        
        if hip_center is None:
            return False
        
        # Calculate body lean vector
        lean_vector = nose - hip_center
        
        # Vector toward opponent
        to_opponent = opponent_pos - hip_center
        to_opponent_norm = to_opponent / np.linalg.norm(to_opponent)
        
        # Project lean onto opponent direction
        lean_toward_opponent = np.dot(lean_vector, to_opponent_norm)
        
        return lean_toward_opponent > 15  # Significant lean toward opponent

    def get_hip_center(self, keypoints, confidences):
        """Get center point between hips"""
        left_hip = keypoints[self.KEYPOINTS['left_hip']]
        right_hip = keypoints[self.KEYPOINTS['right_hip']]
        
        if (confidences[self.KEYPOINTS['left_hip']] > self.CONFIDENCE_THRESHOLD and
            confidences[self.KEYPOINTS['right_hip']] > self.CONFIDENCE_THRESHOLD):
            return (left_hip + right_hip) / 2
        elif confidences[self.KEYPOINTS['left_hip']] > self.CONFIDENCE_THRESHOLD:
            return left_hip
        elif confidences[self.KEYPOINTS['right_hip']] > self.CONFIDENCE_THRESHOLD:
            return right_hip
        return None

    def detect_wide_stance_advanced(self, keypoints, confidences):
        """Enhanced wide stance detection"""
        left_ankle = keypoints[self.KEYPOINTS['left_ankle']]
        right_ankle = keypoints[self.KEYPOINTS['right_ankle']]
        
        if (confidences[self.KEYPOINTS['left_ankle']] > self.CONFIDENCE_THRESHOLD and
            confidences[self.KEYPOINTS['right_ankle']] > self.CONFIDENCE_THRESHOLD):
            
            ankle_distance = abs(left_ankle[0] - right_ankle[0])
            
            # Also consider shoulder width for normalization
            left_shoulder = keypoints[self.KEYPOINTS['left_shoulder']]
            right_shoulder = keypoints[self.KEYPOINTS['right_shoulder']]
            
            if (confidences[self.KEYPOINTS['left_shoulder']] > self.CONFIDENCE_THRESHOLD and
                confidences[self.KEYPOINTS['right_shoulder']] > self.CONFIDENCE_THRESHOLD):
                
                shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
                stance_ratio = ankle_distance / max(shoulder_width, 1)
                
                return stance_ratio > 1.8  # Ankles much wider than shoulders
            else:
                return ankle_distance > 120  # Fallback to absolute distance
        
        return False

    def detect_pointing_gesture(self, keypoints, confidences, opponent_pos):
        """Detect pointing or threatening gestures toward opponent"""
        for side in ['left', 'right']:
            shoulder = keypoints[self.KEYPOINTS[f'{side}_shoulder']]
            elbow = keypoints[self.KEYPOINTS[f'{side}_elbow']]
            wrist = keypoints[self.KEYPOINTS[f'{side}_wrist']]
            
            if (confidences[self.KEYPOINTS[f'{side}_shoulder']] > 0.7 and
                confidences[self.KEYPOINTS[f'{side}_elbow']] > 0.7 and
                confidences[self.KEYPOINTS[f'{side}_wrist']] > 0.7):
                
                # Vector from elbow to wrist (forearm direction)
                forearm_vector = wrist - elbow
                # Vector from wrist to opponent
                to_opponent = opponent_pos - wrist
                
                if np.linalg.norm(forearm_vector) > 0 and np.linalg.norm(to_opponent) > 0:
                    forearm_norm = forearm_vector / np.linalg.norm(forearm_vector)
                    to_opponent_norm = to_opponent / np.linalg.norm(to_opponent)
                    
                    # Check alignment between forearm and direction to opponent
                    alignment = np.dot(forearm_norm, to_opponent_norm)
                    
                    # Also check if arm is extended (not bent back)
                    if alignment > 0.7 and wrist[1] < shoulder[1]:  # Pointing and raised
                        return True
        
        return False

    def detect_clenched_posture(self, keypoints, confidences):
        """Detect tense/clenched body posture"""
        # Check for arms close to body (defensive/aggressive stance)
        left_shoulder = keypoints[self.KEYPOINTS['left_shoulder']]
        right_shoulder = keypoints[self.KEYPOINTS['right_shoulder']]
        left_elbow = keypoints[self.KEYPOINTS['left_elbow']]
        right_elbow = keypoints[self.KEYPOINTS['right_elbow']]
        
        clenched_indicators = 0
        
        # Check if elbows are close to torso
        for side, shoulder, elbow in [('left', left_shoulder, left_elbow), ('right', right_shoulder, right_elbow)]:
            if (confidences[self.KEYPOINTS[f'{side}_shoulder']] > self.CONFIDENCE_THRESHOLD and
                confidences[self.KEYPOINTS[f'{side}_elbow']] > self.CONFIDENCE_THRESHOLD):
                
                elbow_to_shoulder = np.linalg.norm(elbow - shoulder)
                if elbow_to_shoulder < 60:  # Elbow very close to shoulder
                    clenched_indicators += 1
        
        return clenched_indicators >= 1

    def comprehensive_violence_analysis(self, persons, spatial_relations, movement_patterns, frame_shape):
        """Comprehensive multi-person violence analysis"""
        max_risk_score = 0.0
        violence_indicators = []
        
        if len(persons) < self.minimum_people_for_violence:
            return {
                'max_risk_score': 0.0,
                'violence_detected': False,
                'confidence_level': 0.0,
                'violence_indicators': [],
                'person_count': len(persons),
                'recommendation': f'Monitoring - Need {self.minimum_people_for_violence}+ people for violence detection'
            }
        
        # Analyze all person pairs
        for i, person1 in enumerate(persons):
            for j, person2 in enumerate(persons[i+1:], i+1):
                pair_key = f"{i}-{j}"
                
                if pair_key in spatial_relations:
                    relation = spatial_relations[pair_key]
                    # Base risk from spatial relationship
                    spatial_risk = relation['interaction_risk']
                    
                    # Individual pose risks
                    pose1_risk = person1.get('aggressive_features', {}).get('total_aggression_score', 0.0)
                    pose2_risk = person2.get('aggressive_features', {}).get('total_aggression_score', 0.0)
                    
                    # Movement pattern risks
                    movement1_risk = movement_patterns.get(i, {}).get('risk_score', 0.0)
                    movement2_risk = movement_patterns.get(j, {}).get('risk_score', 0.0)
                    
                    # Temporal consistency
                    consistency1 = self.tracker.get_pose_consistency(i)
                    consistency2 = self.tracker.get_pose_consistency(j)
                    temporal_risk = (consistency1 + consistency2) / 2
                    
                    # Combined risk calculation with weights
                    total_risk = (
                        spatial_risk * 0.3 +
                        max(pose1_risk, pose2_risk) * 0.4 +
                        max(movement1_risk, movement2_risk) * 0.2 +
                        temporal_risk * 0.1
                    )
                    
                    # Bonus for mutual aggression
                    if pose1_risk > 0.5 and pose2_risk > 0.5:
                        total_risk += 0.2
                    
                    total_risk = min(total_risk, 1.0)
                    
                    if total_risk > max_risk_score:
                        max_risk_score = total_risk
                    
                    # Record significant interactions
                    if total_risk > 0.4:
                        violence_indicators.append({
                            'persons': [i, j],
                            'risk_score': total_risk,
                            'spatial_risk': spatial_risk,
                            'pose_risks': [pose1_risk, pose2_risk],
                            'movement_risks': [movement1_risk, movement2_risk],
                            'temporal_risk': temporal_risk,
                            'primary_factors': self.identify_risk_factors(person1, person2, relation)
                        })
        
        return {
            'max_risk_score': max_risk_score,
            'violence_detected': max_risk_score > 0.7,
            'confidence_level': min(max_risk_score * 1.1, 1.0),
            'violence_indicators': violence_indicators,
            'person_count': len(persons),
            'recommendation': self.get_recommendation(max_risk_score)
        }

    def identify_risk_factors(self, person1, person2, spatial_relation):
        """Identify primary risk factors for a person pair"""
        factors = []
        
        # Spatial factors
        if spatial_relation['zone'] == 'intimate':
            factors.append('Very close proximity')
        if spatial_relation['facing_each_other']:
            factors.append('Confrontational positioning')
        
        # Pose factors
        for i, person in enumerate([person1, person2], 1):
            pose_features = person.get('aggressive_features', {})
            if pose_features.get('raised_arms', 0) > 1:
                factors.append(f'Person {i}: Raised arms')
            if pose_features.get('pointing_gesture', False):
                factors.append(f'Person {i}: Threatening gesture')
            if pose_features.get('forward_lean', False):
                factors.append(f'Person {i}: Aggressive posture')
        
        return factors

    def get_recommendation(self, risk_score):
        """Get recommendation based on risk score"""
        if risk_score > 0.8:
            return 'IMMEDIATE INTERVENTION REQUIRED'
        elif risk_score > 0.6:
            return 'HIGH ALERT - Monitor closely'
        elif risk_score > 0.4:
            return 'ELEVATED RISK - Increased surveillance'
        elif risk_score > 0.2:
            return 'LOW RISK - Continue monitoring'
        else:
            return 'NORMAL - Routine surveillance'

    def find_closest_person(self, person, all_persons, exclude_index):
        """Find the closest person to the given person"""
        min_distance = float('inf')
        closest_person = None
        
        for i, other_person in enumerate(all_persons):
            if i != exclude_index and other_person['center'] is not None:
                distance = np.linalg.norm(person['center'] - other_person['center'])
                if distance < min_distance:
                    min_distance = distance
                    closest_person = other_person
        
        return closest_person

    def draw_multi_person_annotations(self, frame, persons, spatial_relations, analysis_result):
        """Draw comprehensive annotations for multi-person analysis"""
        try:
            # Draw person skeletons and keypoints
            for person in persons:
                # Color based on individual aggression level
                aggression_score = person.get('aggressive_features', {}).get('total_aggression_score', 0.0)
                if aggression_score > 0.6:
                    color = (0, 0, 255)  # Red for high aggression
                elif aggression_score > 0.3:
                    color = (0, 165, 255)  # Orange for medium aggression
                else:
                    color = (0, 255, 0)  # Green for low/no aggression
                
                frame = self.draw_skeleton(frame, person['keypoints'], person['confidences'], color, 2)
                frame = self.draw_keypoints(frame, person['keypoints'], person['confidences'], color, 4)
                
                # Draw person ID and aggression score
                center = person['center'].astype(int)
                cv2.putText(frame, f"P{person['id']}: {aggression_score:.2f}", 
                           (center[0]-20, center[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw interaction lines
            for pair_key, relation in spatial_relations.items():
                person_ids = [int(x) for x in pair_key.split('-')]
                if len(person_ids) == 2:
                    p1_center = persons[person_ids[0]]['center'].astype(int)
                    p2_center = persons[person_ids[1]]['center'].astype(int)
                    
                    # Line color based on interaction risk
                    risk = relation['interaction_risk']
                    if risk > 0.5:
                        line_color = (0, 0, 255)  # Red
                    elif risk > 0.3:
                        line_color = (0, 165, 255)  # Orange
                    else:
                        line_color = (255, 255, 0)  # Yellow
                    
                    cv2.line(frame, tuple(p1_center), tuple(p2_center), line_color, 2)
                    
                    # Draw zone and distance info
                    mid_point = ((p1_center + p2_center) // 2).astype(int)
                    cv2.putText(frame, f"{relation['zone']}", 
                               (mid_point[0], mid_point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, line_color, 1)
            
            # Overall status
            status_color = (0, 0, 255) if analysis_result['violence_detected'] else (0, 255, 0)
            status_text = f"VIOLENCE: {'DETECTED' if analysis_result['violence_detected'] else 'NORMAL'}"
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            # Risk score and person count
            cv2.putText(frame, f"Risk: {analysis_result['max_risk_score']:.2f}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            cv2.putText(frame, f"People: {analysis_result['person_count']}", 
                       (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Recommendation
            cv2.putText(frame, analysis_result['recommendation'], 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
        except Exception as e:
            logger.error(f"Error drawing multi-person annotations: {e}")
        
        return frame

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

    def detect_raised_arms(self, keypoints, confidences):
        """Legacy method for backward compatibility"""
        return int(self.detect_raised_arms_advanced(keypoints, confidences))

    def detect_aggressive_pose(self, keypoints, confidences):
        """Legacy method for backward compatibility"""
        return self.detect_wide_stance_advanced(keypoints, confidences)

    def analyze_keypoints(self, keypoints, confidences, image_shape, person_count=1):
        """Legacy analyze keypoints method for backward compatibility"""
        # This method is kept for compatibility but the main analysis
        # now happens in comprehensive_violence_analysis
        features = {
            'raised_arms': self.detect_raised_arms(keypoints, confidences),
            'aggressive_pose': self.detect_aggressive_pose(keypoints, confidences),
            'person_count': person_count,
            'sufficient_people': person_count >= self.minimum_people_for_violence,
            'detection_state': "NORMAL",
            'detection_probability': 0.0,
            'detection_type': 'violence'
        }

        if not features['sufficient_people']:
            features['detection_state'] = f"MONITORING ({person_count} PERSON{'S' if person_count != 1 else ''})"
            features['detection_probability'] = 0.0
            return features

        # Simple violence detection logic for legacy compatibility
        if features['raised_arms'] >= 2 and features['aggressive_pose']:
            features['detection_state'] = "VIOLENCE (HIGH RISK)"
            features['detection_probability'] = 0.9
        elif features['raised_arms'] >= 1 and features['aggressive_pose']:
            features['detection_state'] = "VIOLENCE (MEDIUM RISK)"
            features['detection_probability'] = 0.7
        elif features['raised_arms'] >= 2:
            features['detection_state'] = "AGGRESSION DETECTED"
            features['detection_probability'] = 0.6
        elif features['aggressive_pose']:
            features['detection_state'] = "SUSPICIOUS BEHAVIOR"
            features['detection_probability'] = 0.4

        return features

    def draw_annotations(self, frame, keypoints, confidences, features, confidence_threshold):
        """Legacy draw annotations method"""
        try:
            frame = self.draw_skeleton(frame, keypoints, confidences, color=(0, 255, 0), thickness=2)
            frame = self.draw_keypoints(frame, keypoints, confidences, color=(0, 255, 0), radius=5)
            
            person_count_text = f"People detected: {features['person_count']}"
            count_color = (0, 255, 0) if features['sufficient_people'] else (0, 165, 255)
            cv2.putText(frame, person_count_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, count_color, 2)
            
            if not features['sufficient_people']:
                requirement_text = f"Need {self.minimum_people_for_violence}+ people for violence detection"
                cv2.putText(frame, requirement_text, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            if features['sufficient_people']:
                y_offset = 85
                if features['raised_arms'] > 0:
                    arms_text = f"Raised Arms: {features['raised_arms']}"
                    cv2.putText(frame, arms_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    y_offset += 25
                
                if features['aggressive_pose']:
                    cv2.putText(frame, "AGGRESSIVE STANCE", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            color = (0, 0, 255) if features['detection_probability'] > 0.7 else (0, 255, 0)
            cv2.putText(frame, f"VIOLENCE: {features['detection_state']}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
        except Exception as e:
            logger.error(f"Error drawing violence annotations: {e}")
        
        return frame

    def analyze(self, frame, confidence, alert_threshold, finger_tips=None):
        """
        Enhanced multi-person violence analysis
        
        Args:
            frame: Input frame for analysis
            confidence: Confidence threshold for pose detection
            alert_threshold: Threshold for triggering alerts
            finger_tips: Optional finger tip coordinates (not used in violence detection)
            
        Returns:
            tuple: (annotated_frame, detection_status)
        """
        try:
            results = self.model(frame, conf=confidence, verbose=False)
            annotated_frame = frame.copy()
            
            # Step 1: Detect multiple persons
            persons = self.detect_multiple_persons(results)
            
            if len(persons) < self.minimum_people_for_violence:
                # Show monitoring status when insufficient people
                person_count = len(persons)
                cv2.putText(annotated_frame, f"People detected: {person_count}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                cv2.putText(annotated_frame, f"Need {self.minimum_people_for_violence}+ people for violence detection", 
                           (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                # cv2.putText(annotated_frame, f"VIOLENCE: MONITORING ({person_count} PERSON{'S' if person_count != 1 else ''})", 
                #            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Still draw detected persons
                for person in persons:
                    annotated_frame = self.draw_skeleton(annotated_frame, person['keypoints'], 
                                                       person['confidences'], color=(0, 255, 0), thickness=2)
                    annotated_frame = self.draw_keypoints(annotated_frame, person['keypoints'], 
                                                        person['confidences'], color=(0, 255, 0), radius=4)
                
                detection_status = {
                    'type': 'violence',
                    'detected': False,
                    'probability': 0.0,
                    'state': f'MONITORING ({person_count} PERSONS)'
                }
                return annotated_frame, detection_status
            
            # Step 2: Analyze spatial relationships between all person pairs
            spatial_relations = {}
            for i, person1 in enumerate(persons):
                for j, person2 in enumerate(persons[i+1:], i+1):
                    relation = self.analyze_spatial_relationship(person1, person2, frame.shape[:2])
                    if relation:
                        spatial_relations[f"{i}-{j}"] = relation
            
            # Step 3: Analyze individual aggressive poses
            for i, person in enumerate(persons):
                closest_opponent = self.find_closest_person(person, persons, i)
                opponent_pos = closest_opponent['center'] if closest_opponent else None
                
                aggressive_features = self.analyze_aggressive_pose_advanced(person, opponent_pos)
                person['aggressive_features'] = aggressive_features
                
                # Update tracker for temporal analysis
                tracker_features = {
                    'center': person['center'],
                    'aggressive_features': aggressive_features
                }
                self.tracker.update_person_history(i, tracker_features)
            
            # Step 4: Analyze movement patterns
            movement_patterns = {}
            for i, person in enumerate(persons):
                movement_patterns[i] = self.tracker.analyze_movement_pattern(i)
            
            # Step 5: Comprehensive violence analysis
            analysis_result = self.comprehensive_violence_analysis(
                persons, spatial_relations, movement_patterns, frame.shape[:2]
            )
            
            # Step 6: Draw comprehensive annotations
            annotated_frame = self.draw_multi_person_annotations(
                annotated_frame, persons, spatial_relations, analysis_result
            )
            
            # Prepare detection status
            detection_status = {
                'type': 'violence',
                'detected': analysis_result['violence_detected'],
                'probability': analysis_result['max_risk_score'],
                'state': 'VIOLENCE DETECTED' if analysis_result['violence_detected'] else 'NORMAL',
                'confidence_level': analysis_result['confidence_level'],
                'person_count': analysis_result['person_count'],
                'violence_indicators': analysis_result['violence_indicators'],
                'recommendation': analysis_result['recommendation']
            }
            
            return annotated_frame, detection_status
            
        except Exception as e:
            logger.error(f"Error in enhanced violence analysis: {e}")
            detection_status = {
                'type': 'violence',
                'detected': False,
                'probability': 0.0,
                'state': 'ERROR',
                'error': str(e)
            }
            return frame, detection_status