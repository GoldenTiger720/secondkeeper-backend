from rest_framework import viewsets, generics, status, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from django.utils import timezone
import cv2
import logging

from .models import Camera
from .serializers import (
    CameraSerializer, CameraCreateSerializer, CameraUpdateSerializer,
    CameraStatusSerializer, CameraListSerializer, CameraSettingsSerializer
)
from utils.permissions import IsOwnerOrAdmin, IsManagerOrAdminOrReviewer
from utils.stream_proxy import start_camera_stream, stop_all_streams
logger = logging.getLogger('security_ai')

class CameraViewSet(viewsets.ModelViewSet):
    """ViewSet for managing cameras."""
    
    permission_classes = [permissions.IsAuthenticated, IsOwnerOrAdmin]
    serializer_class = CameraSerializer
    
    def get_queryset(self):
        """Return all cameras for admins, or just the user's cameras for regular users."""
        user = self.request.user
        if user.is_admin():
            return Camera.objects.all()
        return Camera.objects.filter(user=user)
    
    def get_serializer_class(self):
        """Return appropriate serializer class based on the action."""
        if self.action == 'create':
            return CameraCreateSerializer
        elif self.action == 'update' or self.action == 'partial_update':
            return CameraUpdateSerializer
        elif self.action == 'list':
            return CameraListSerializer
        elif self.action == 'update_status':
            return CameraStatusSerializer
        elif self.action == 'settings':
            return CameraSettingsSerializer
        return self.serializer_class
    
    def create(self, request, *args, **kwargs):
        """Create a new camera."""
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        camera = serializer.save()

        connection_status = self._check_camera_connection(camera)
        camera.status = connection_status
        if connection_status == 'online':
            camera.last_online = timezone.now()
        camera.save(update_fields=['status', 'last_online', 'updated_at'])
        headers = self.get_success_headers(serializer.data)
        return Response({
            'success': True,
            'data': CameraSerializer(camera).data,
            'message': 'Camera created successfully.',
            'errors': []
        }, status=status.HTTP_201_CREATED, headers=headers)
    
    def update(self, request, *args, **kwargs):
        """Update a camera."""
        partial = kwargs.pop('partial', False)
        instance = self.get_object()
        serializer = self.get_serializer(instance, data=request.data, partial=partial)
        serializer.is_valid(raise_exception=True)
        camera = serializer.save()
        
        return Response({
            'success': True,
            'data': CameraSerializer(camera).data,
            'message': 'Camera updated successfully.',
            'errors': []
        })
    
    def retrieve(self, request, *args, **kwargs):
        """Retrieve a camera."""
        instance = self.get_object()
        serializer = CameraSerializer(instance)
        
        return Response({
            'success': True,
            'data': serializer.data,
            'message': 'Camera retrieved successfully.',
            'errors': []
        })
    
    def list(self, request, *args, **kwargs):
        """List cameras."""
        queryset = self.filter_queryset(self.get_queryset())
        page = self.paginate_queryset(queryset)
        
        if page is not None:
            serializer = self.get_serializer(page, many=True)
            response = self.get_paginated_response(serializer.data)
            response.data = {
                'success': True,
                'data': response.data,
                'message': 'Cameras retrieved successfully.',
                'errors': []
            }
            return response
        
        serializer = self.get_serializer(queryset, many=True)
        return Response({
            'success': True,
            'data': serializer.data,
            'message': 'Cameras retrieved successfully.',
            'errors': []
        })
    
    def destroy(self, request, *args, **kwargs):
        """Delete a camera."""
        instance = self.get_object()
        camera_id = instance.id
        self.perform_destroy(instance)
        
        return Response({
            'success': True,
            'data': {},
            'message': 'Camera deleted successfully.',
            'errors': []
        }, status=status.HTTP_200_OK)

    def _check_camera_connection(self, camera):
        try:
            stream_url = camera.get_stream_url()
            cap = cv2.VideoCapture(stream_url)
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                if ret:
                    logger.info(f"Camera connection successful: {camera.id} - {camera.name}")
                    return 'online'
                else:
                    logger.warning(f"Camera connected but failed to read frame: {camera.id} - {camera.name}")
                    return 'error'
            else:
                logger.warning(f"Failed to connect to camera: {camera.id} - {camera.name}")
                return 'offline'
        except Exception as e:
            logger.error(f"Error checking camera connection: {camera.id} - {camera.name} - {str(e)}")
            return 'error'
    
    @action(detail=True, methods=['get'], permission_classes=[])
    def stream(self, request, pk=None):
        """Get camera livestream URL for HLS streaming or direct MP4 streaming for RTSP."""
        try:
            # Handle token authentication from URL parameter
            token = request.query_params.get('token')
            auth_header = request.headers.get('Authorization')
            
            # If user is not authenticated, try token from URL parameter or header
            if not request.user.is_authenticated:
                from rest_framework_simplejwt.authentication import JWTAuthentication
                from rest_framework_simplejwt.exceptions import InvalidToken, TokenError
                
                jwt_token = None
                
                # Try URL parameter first, then Authorization header
                if token:
                    jwt_token = token
                elif auth_header and auth_header.startswith('Bearer '):
                    jwt_token = auth_header.split(' ')[1]
                
                if jwt_token:
                    try:
                        jwt_auth = JWTAuthentication()
                        validated_token = jwt_auth.get_validated_token(jwt_token)
                        user = jwt_auth.get_user(validated_token)
                        
                        # Set authenticated user for this request
                        request.user = user
                        request.auth = validated_token
                        
                    except (InvalidToken, TokenError) as e:
                        return Response({
                            'success': False,
                            'data': {},
                            'message': f'Invalid or expired token: {str(e)}',
                            'errors': ['Authentication failed']
                        }, status=status.HTTP_401_UNAUTHORIZED)
                else:
                    return Response({
                        'success': False,
                        'data': {},
                        'message': 'Authentication required',
                        'errors': ['No token provided']
                    }, status=status.HTTP_401_UNAUTHORIZED)
            
            # Check permissions (admin, manager, or reviewer roles only)
            if not hasattr(request.user, 'role') or request.user.role not in ['admin', 'manager', 'reviewer']:
                return Response({
                    'success': False,
                    'data': {},
                    'message': 'Permission denied',
                    'errors': ['Insufficient permissions - admin, manager, or reviewer role required']
                }, status=status.HTTP_403_FORBIDDEN)
            
            camera = self.get_object()
            quality = request.query_params.get('quality', 'medium')
            print(camera.id, camera.name, camera.status)
            
            # Check if camera URL is RTSP for direct streaming
            if camera.stream_url.startswith('rtsp://'):
                # Import the RTSP streaming function
                from utils.rtsp_stream import create_rtsp_streaming_response
                
                # Return direct streaming response
                return create_rtsp_streaming_response(camera)
            
            # For non-RTSP URLs, use HLS streaming
            # Create and start stream proxy
            result = start_camera_stream(camera, target_fps=15)
            
            if result['success'] and result['hls_ready']:
                return Response({
                    'success': True,
                    'data': {
                        'stream_url': result['stream_url'],
                        'quality': quality,
                        'camera_info': {
                            'id': camera.id,
                            'name': camera.name,
                            'status': camera.status
                        }
                    },
                    'message': result['message'],
                    'errors': []
                })
            else:
                return Response({
                    'success': False,
                    'data': {},
                    'message': result['message'],
                    'errors': [result['message']]
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
        except Camera.DoesNotExist:
            return Response({
                'success': False,
                'data': {},
                'message': 'Camera not found',
                'errors': ['Invalid camera ID']
            }, status=status.HTTP_404_NOT_FOUND)
        
        except Exception as e:
            logger.error(f"Stream URL retrieval error: {str(e)}")
            return Response({
                'success': False,
                'data': {},
                'message': 'Error retrieving stream URL',
                'errors': [str(e)]
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    @action(detail=False, methods=['get'])
    def status(self, request):
        """Get the status of all cameras."""
        cameras = self.get_queryset()
        serializer = CameraStatusSerializer(cameras, many=True)
        
        return Response({
            'success': True,
            'data': serializer.data,
            'message': 'Camera statuses retrieved successfully.',
            'errors': []
        })
    
    def stop_all_streams_view(request):
        stop_all_streams()
        return JsonResponse({
            'success': True,
            'message': 'All streams stopped'
        })