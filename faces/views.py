from rest_framework import viewsets, status, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from django.db.models import Q
from rest_framework.views import APIView
import numpy as np
import cv2
from PIL import Image
import io
import logging
import os
import pickle

from .models import AuthorizedFace, FaceVerificationLog
from .serializers import (
    AuthorizedFaceSerializer, AuthorizedFaceCreateSerializer,
    AuthorizedFaceUpdateSerializer, FaceVerificationSerializer,
    FaceVerificationRequestSerializer, FaceVerificationResponseSerializer, FaceUploadSerializer
)
from cameras.models import Camera
from utils.permissions import IsOwnerOrAdmin

logger = logging.getLogger('security_ai')

class AuthorizedFaceViewSet(viewsets.ModelViewSet):
    """ViewSet for managing authorized faces."""
    
    permission_classes = [permissions.IsAuthenticated, IsOwnerOrAdmin]
    serializer_class = AuthorizedFaceSerializer
    
    def get_queryset(self):
        """Return all authorized faces for admins, or just the user's faces for regular users."""
        user = self.request.user
        if user.is_admin():
            return AuthorizedFace.objects.all()
        return AuthorizedFace.objects.filter(user=user)
    
    def get_serializer_class(self):
        """Return appropriate serializer class based on the action."""
        if self.action == 'create':
            return AuthorizedFaceCreateSerializer
        elif self.action in ['update', 'partial_update']:
            return AuthorizedFaceUpdateSerializer
        return self.serializer_class
    
    def perform_create(self, serializer):
        """Save the authorized face and generate face encoding."""
        face = serializer.save()
    
    def perform_update(self, serializer):
        """Update the authorized face and regenerate face encoding if image changed."""
        face = serializer.save()
    
    def create(self, request, *args, **kwargs):
        """Create a new authorized face."""
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        try:
            self.perform_create(serializer)
            headers = self.get_success_headers(serializer.data)
            
            return Response({
                'success': True,
                'data': serializer.data,
                'message': 'Authorized face created successfully.',
                'errors': []
            }, status=status.HTTP_201_CREATED, headers=headers)
            
        except Exception as e:
            logger.error(f"Error creating authorized face: {str(e)}")
            return Response({
                'success': False,
                'data': {},
                'message': 'Failed to create authorized face.',
                'errors': [str(e)]
            }, status=status.HTTP_400_BAD_REQUEST)
    
    def update(self, request, *args, **kwargs):
        """Update an authorized face."""
        partial = kwargs.pop('partial', False)
        instance = self.get_object()
        serializer = self.get_serializer(instance, data=request.data, partial=partial)
        serializer.is_valid(raise_exception=True)
        
        try:
            self.perform_update(serializer)
            
            return Response({
                'success': True,
                'data': serializer.data,
                'message': 'Authorized face updated successfully.',
                'errors': []
            })
            
        except Exception as e:
            logger.error(f"Error updating authorized face: {str(e)}")
            return Response({
                'success': False,
                'data': {},
                'message': 'Failed to update authorized face.',
                'errors': [str(e)]
            }, status=status.HTTP_400_BAD_REQUEST)
    
    def retrieve(self, request, *args, **kwargs):
        """Retrieve an authorized face."""
        instance = self.get_object()
        serializer = self.get_serializer(instance)
        
        return Response({
            'success': True,
            'data': serializer.data,
            'message': 'Authorized face retrieved successfully.',
            'errors': []
        })
    
    def list(self, request, *args, **kwargs):
        """List authorized faces."""
        queryset = self.filter_queryset(self.get_queryset())
        
        # Apply filters
        name_filter = request.query_params.get('name')
        if name_filter:
            queryset = queryset.filter(name__icontains=name_filter)
        
        role_filter = request.query_params.get('role')
        if role_filter:
            queryset = queryset.filter(role__icontains=role_filter)
        
        active_filter = request.query_params.get('is_active')
        if active_filter is not None:
            is_active = active_filter.lower() == 'true'
            queryset = queryset.filter(is_active=is_active)
        
        # Handle pagination
        page = self.paginate_queryset(queryset)
        if page is not None:
            serializer = self.get_serializer(page, many=True)
            response = self.get_paginated_response(serializer.data)
            response.data = {
                'success': True,
                'data': response.data,
                'message': 'Authorized faces retrieved successfully.',
                'errors': []
            }
            return response
        
        serializer = self.get_serializer(queryset, many=True)
        return Response({
            'success': True,
            'data': serializer.data,
            'message': 'Authorized faces retrieved successfully.',
            'errors': []
        })
    
    def destroy(self, request, *args, **kwargs):
        """Delete an authorized face."""
        instance = self.get_object()
        
        try:
            # Delete the face image file if it exists
            if instance.face_image:
                try:
                    if os.path.isfile(instance.face_image.path):
                        os.remove(instance.face_image.path)
                except Exception as e:
                    logger.warning(f"Could not delete face image file: {str(e)}")
            
            self.perform_destroy(instance)
            
            return Response({
                'success': True,
                'data': {},
                'message': 'Authorized face deleted successfully.',
                'errors': []
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error deleting authorized face: {str(e)}")
            return Response({
                'success': False,
                'data': {},
                'message': 'Failed to delete authorized face.',
                'errors': [str(e)]
            }, status=status.HTTP_400_BAD_REQUEST)
    
    @action(detail=False, methods=['post'])
    def verify(self, request):
        """Verify if a face matches an authorized person."""
        serializer = FaceVerificationRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        face_image = serializer.validated_data['face_image']
        camera_id = serializer.validated_data.get('camera_id')
        confidence_threshold = serializer.validated_data.get('confidence_threshold', 0.6)
        
        camera = None
        if camera_id:
            try:
                camera = Camera.objects.get(id=camera_id)
                # Check if the user owns the camera or is an admin
                if camera.user != request.user and not request.user.is_admin():
                    return Response({
                        'success': False,
                        'data': {},
                        'message': 'You do not have permission to access this camera.',
                        'errors': ['Permission denied.']
                    }, status=status.HTTP_403_FORBIDDEN)
            except Camera.DoesNotExist:
                return Response({
                    'success': False,
                    'data': {},
                    'message': 'Camera not found.',
                    'errors': ['Invalid camera ID.']
                }, status=status.HTTP_404_NOT_FOUND)
        
        try:
            # Process the uploaded face image
            image = Image.open(face_image)
            image_array = np.array(image)
            
            # Convert to grayscale
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_array
            
            # Detect faces
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                # Create verification log
                verification_log = FaceVerificationLog.objects.create(
                    is_match=False,
                    confidence=0.0,
                    source_image=face_image,
                    source_camera=camera,
                    notes="No face detected in the image."
                )
                
                return Response({
                    'success': False,
                    'data': {
                        'is_match': False,
                        'confidence': 0.0,
                        'matched_face': None,
                        'verification_id': verification_log.id
                    },
                    'message': 'No face detected in the image.',
                    'errors': ['No face detected.']
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Use the largest face if multiple faces detected
            x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
            face_roi = gray[y:y+h, x:x+w]
            
            # Resize and normalize
            face_roi_resized = cv2.resize(face_roi, (128, 128))
            face_array = face_roi_resized.flatten() / 255.0
            
            # Compare with stored face encodings
            best_match = None
            highest_confidence = 0.0
            
            # Get all active authorized faces for the user (or all if admin)
            if request.user.is_admin():
                authorized_faces = AuthorizedFace.objects.filter(is_active=True)
            else:
                authorized_faces = AuthorizedFace.objects.filter(user=request.user, is_active=True)
            
            for auth_face in authorized_faces:
                if auth_face.face_encoding:
                    try:
                        # Deserialize the stored encoding
                        stored_encoding = pickle.loads(auth_face.face_encoding)
                        
                        # Compute similarity (using cosine similarity)
                        similarity = np.dot(face_array, stored_encoding) / (
                            np.linalg.norm(face_array) * np.linalg.norm(stored_encoding)
                        )
                        
                        # Normalize similarity to 0-1 range as confidence
                        confidence = (similarity + 1) / 2
                        
                        if confidence > highest_confidence:
                            highest_confidence = confidence
                            best_match = auth_face
                    except Exception as e:
                        logger.error(f"Error comparing face encodings: {str(e)}")
                        continue
            
            # Create verification log
            verification_log = FaceVerificationLog.objects.create(
                authorized_face=best_match if highest_confidence >= confidence_threshold else None,
                is_match=highest_confidence >= confidence_threshold,
                confidence=highest_confidence,
                source_image=face_image,
                source_camera=camera
            )
            
            # Return the verification result
            response_data = {
                'is_match': highest_confidence >= confidence_threshold,
                'confidence': highest_confidence,
                'matched_face': AuthorizedFaceSerializer(best_match).data if best_match and highest_confidence >= confidence_threshold else None,
                'verification_id': verification_log.id
            }
            
            message = 'Face verification completed.'
            if highest_confidence >= confidence_threshold:
                message = f'Face matched with {best_match.name} (confidence: {highest_confidence:.2f}).'
            else:
                message = 'No matching face found.'
            
            return Response({
                'success': True,
                'data': response_data,
                'message': message,
                'errors': []
            })
            
        except Exception as e:
            logger.error(f"Error in face verification: {str(e)}")
            return Response({
                'success': False,
                'data': {},
                'message': 'Error processing face verification.',
                'errors': [str(e)]
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    @action(detail=False, methods=['get'])
    def by_role(self, request):
        """Get faces grouped by role."""
        try:
            queryset = self.get_queryset().filter(is_active=True)
            
            # Group faces by role
            faces_by_role = {}
            roles = ['primary', 'caregiver', 'family', 'other']
            
            for role in roles:
                role_faces = queryset.filter(role=role)
                serializer = self.get_serializer(role_faces, many=True)
                faces_by_role[role] = serializer.data
            
            return Response({
                'success': True,
                'data': faces_by_role,
                'message': 'Faces retrieved by role successfully.',
                'errors': []
            })
            
        except Exception as e:
            logger.error(f"Error getting faces by role: {str(e)}")
            return Response({
                'success': False,
                'data': {},
                'message': 'Error retrieving faces by role.',
                'errors': [str(e)]
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class FaceUploadView(APIView):
    """
    API view for uploading face images.
    This provides an alternative endpoint that matches the frontend expectations.
    """
    
    permission_classes = [permissions.IsAuthenticated]
    
    def post(self, request, *args, **kwargs):
        """Handle face upload requests."""
        serializer = FaceUploadSerializer(data=request.data)
        
        if not serializer.is_valid():
            return Response({
                'success': False,
                'data': {},
                'message': 'Validation failed.',
                'errors': [str(error) for field_errors in serializer.errors.values() for error in field_errors]
            }, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            # Create the authorized face
            face_data = {
                'name': serializer.validated_data['name'],
                'role': serializer.validated_data['role'],
                'face_image': serializer.validated_data['face_image'],
                'description': serializer.validated_data.get('description', ''),
                'user': request.user,
                'is_active': True
            }
            
            # Create the face instance
            authorized_face = AuthorizedFace.objects.create(**face_data)
            
            # Serialize the created face
            face_serializer = AuthorizedFaceSerializer(authorized_face, context={'request': request})
            
            return Response({
                'success': True,
                'data': face_serializer.data,
                'message': 'Face uploaded and processed successfully.',
                'errors': []
            }, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            logger.error(f"Error uploading face: {str(e)}")
            return Response({
                'success': False,
                'data': {},
                'message': 'Failed to upload face.',
                'errors': [str(e)]
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)