# detectors/face_detector.py

import cv2
import numpy as np
import pickle
import logging
from django.db.models import Q
from datetime import datetime
from faces.models import AuthorizedFace, FaceVerificationLog
import tempfile
import os

logger = logging.getLogger('security_ai')

class FaceDetector:
    """Detector for face recognition and unauthorized face detection"""
    
    def __init__(self):
        """Initialize the face detector"""
        self.name = "Face Recognition"
        self.face_cascade = None
        self.load_face_detector()
        
    def load_face_detector(self):
        """Load OpenCV face detection cascade"""
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            if self.face_cascade.empty():
                logger.error("Failed to load face cascade classifier")
                self.face_cascade = None
            else:
                logger.info("Face cascade classifier loaded successfully")
        except Exception as e:
            logger.error(f"Error loading face cascade: {str(e)}")
            self.face_cascade = None
    
    def detect_faces_in_frame(self, frame, camera, confidence_threshold=0.6):
        """
        Detect and recognize faces in a frame
        
        Args:
            frame: OpenCV frame
            camera: Camera model instance
            confidence_threshold: Threshold for face recognition
            
        Returns:
            tuple: (has_unauthorized_face, max_confidence, annotated_frame, results)
        """
        if self.face_cascade is None:
            return False, 0.0, frame, []
            
        try:
            # Convert frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            if len(faces) == 0:
                return False, 0.0, frame, []
            
            # Get authorized faces for the camera's user
            authorized_faces = AuthorizedFace.objects.filter(
                user=camera.user,
                is_active=True
            )
            
            annotated_frame = frame.copy()
            has_unauthorized = False
            max_confidence = 0.0
            results = []
            
            for (x, y, w, h) in faces:
                # Extract face region
                face_roi = gray[y:y+h, x:x+w]
                face_roi_resized = cv2.resize(face_roi, (128, 128))
                face_array = face_roi_resized.flatten() / 255.0
                
                # Check against authorized faces
                best_match = None
                best_similarity = 0.0
                
                for auth_face in authorized_faces:
                    if auth_face.face_encoding:
                        try:
                            stored_encoding = pickle.loads(auth_face.face_encoding)
                            
                            # Compute similarity
                            similarity = np.dot(face_array, stored_encoding) / (
                                np.linalg.norm(face_array) * np.linalg.norm(stored_encoding)
                            )
                            
                            # Normalize to 0-1 range
                            confidence = (similarity + 1) / 2
                            
                            if confidence > best_similarity:
                                best_similarity = confidence
                                best_match = auth_face
                                
                        except Exception as e:
                            logger.error(f"Error comparing face encodings: {str(e)}")
                            continue
                
                # Determine if face is authorized
                is_authorized = best_similarity >= confidence_threshold
                
                if not is_authorized:
                    has_unauthorized = True
                    max_confidence = max(max_confidence, 1.0 - best_similarity)
                
                # Draw rectangle and label
                color = (0, 255, 0) if is_authorized else (0, 0, 255)
                cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), color, 2)
                
                label = f"{best_match.name} ({best_similarity:.2f})" if is_authorized and best_match else "Unauthorized"
                cv2.putText(annotated_frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Log verification
                try:
                    # Save face image temporarily for logging
                    face_img = frame[y:y+h, x:x+w]
                    temp_path = self._save_temp_face_image(face_img)
                    
                    FaceVerificationLog.objects.create(
                        authorized_face=best_match if is_authorized else None,
                        is_match=is_authorized,
                        confidence=best_similarity,
                        source_camera=camera,
                        notes=f"Automatic detection - {'Authorized' if is_authorized else 'Unauthorized'}"
                    )
                    
                    # Clean up temp file
                    if temp_path and os.path.exists(temp_path):
                        os.remove(temp_path)
                        
                except Exception as e:
                    logger.error(f"Error logging face verification: {str(e)}")
                
                results.append({
                    'bbox': (x, y, w, h),
                    'is_authorized': is_authorized,
                    'confidence': best_similarity,
                    'match': best_match.name if best_match else None
                })
            
            return has_unauthorized, max_confidence, annotated_frame, results
            
        except Exception as e:
            logger.error(f"Error in face detection: {str(e)}")
            return False, 0.0, frame, []
    
    def _save_temp_face_image(self, face_img):
        """Save face image temporarily for logging"""
        try:
            # Create temp file
            temp_fd, temp_path = tempfile.mkstemp(suffix='.jpg')
            os.close(temp_fd)
            
            # Save image
            cv2.imwrite(temp_path, face_img)
            return temp_path
            
        except Exception as e:
            logger.error(f"Error saving temp face image: {str(e)}")
            return None
    
    def get_description(self):
        """Return a description of the detector"""
        return "Detects faces and identifies unauthorized individuals in surveillance footage."
    
    def get_model_info(self):
        """Return information about the model used by the detector"""
        return {
            "name": self.name,
            "type": "OpenCV Haar Cascade + Face Recognition",
            "classes": ["authorized_face", "unauthorized_face"]
        }