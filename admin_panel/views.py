# admin_panel/views.py - Updated with complete role-based access control

from rest_framework import viewsets, generics, status, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from django.contrib.auth import get_user_model
from django.db.models import Count, Q
from django.utils import timezone
import datetime
import psutil
import platform
import os
import logging

from .models import SystemCheck, SystemSetting, SubscriptionPlan, UserSubscription
from .serializers import (
    SystemCheckSerializer, SystemSettingSerializer, 
    SubscriptionPlanSerializer, UserSubscriptionSerializer,
    UserAdminSerializer, UserUpdateAdminSerializer,
    SystemStatusSerializer
)
from cameras.models import Camera
from alerts.models import Alert
from utils.permissions import IsAdminUser, IsManagerOrAdminOrReviewer, CanAddRoles
from accounts.serializers import UserCreateSerializer

User = get_user_model()
logger = logging.getLogger('security_ai')

class UserAdminViewSet(viewsets.ModelViewSet):
    """ViewSet for admin management of users."""
    
    permission_classes = [permissions.IsAuthenticated, IsManagerOrAdminOrReviewer]
    serializer_class = UserAdminSerializer
    
    def get_queryset(self):
        """Get users based on current user's role."""
        user = self.request.user
        
        if user.is_admin():
            # Admins can see all users except other admins
            queryset = User.objects.exclude(role='admin').order_by('-date_joined')
        elif user.is_manager():
            # Managers can only see regular users (clients)
            queryset = User.objects.filter(role='user').order_by('-date_joined')
        else:
            # Reviewers and regular users can't see any users
            queryset = User.objects.none()
        
        # Annotate with camera and alert counts
        queryset = queryset.annotate(
            cameras_count=Count('cameras', distinct=True),
            alerts_count=Count('cameras__alerts', distinct=True)
        )
        
        # Apply filters if provided
        status_filter = self.request.query_params.get('status')
        if status_filter:
            queryset = queryset.filter(status=status_filter)
        
        is_active = self.request.query_params.get('is_active')
        if is_active is not None:
            queryset = queryset.filter(is_active=is_active.lower() == 'true')
        
        search = self.request.query_params.get('search')
        if search:
            queryset = queryset.filter(
                Q(email__icontains=search) |
                Q(full_name__icontains=search) |
                Q(phone_number__icontains=search)
            )
        
        return queryset
    
    def get_serializer_class(self):
        """Return appropriate serializer class based on the action."""
        if self.action in ['update', 'partial_update']:
            return UserUpdateAdminSerializer
        elif self.action == 'add_role':
            return UserCreateSerializer
        return self.serializer_class
    
    def list(self, request, *args, **kwargs):
        """List users based on permissions."""
        queryset = self.filter_queryset(self.get_queryset())
        page = self.paginate_queryset(queryset)
        
        if page is not None:
            serializer = self.get_serializer(page, many=True)
            response = self.get_paginated_response(serializer.data)
            response.data = {
                'success': True,
                'data': response.data,
                'message': 'Users retrieved successfully.',
                'errors': []
            }
            return response
        
        serializer = self.get_serializer(queryset, many=True)
        return Response({
            'success': True,
            'data': serializer.data,
            'message': 'Users retrieved successfully.',
            'errors': []
        })
    
    def retrieve(self, request, *args, **kwargs):
        """Retrieve a user."""
        instance = self.get_object()
        serializer = self.get_serializer(instance)
        
        return Response({
            'success': True,
            'data': serializer.data,
            'message': 'User retrieved successfully.',
            'errors': []
        })
    
    def update(self, request, *args, **kwargs):
        """Update a user."""
        partial = kwargs.pop('partial', False)
        instance = self.get_object()
        serializer = self.get_serializer(instance, data=request.data, partial=partial)
        serializer.is_valid(raise_exception=True)
        self.perform_update(serializer)
        
        # Re-fetch with annotations
        instance = self.get_queryset().get(pk=instance.pk)
        return Response({
            'success': True,
            'data': UserAdminSerializer(instance).data,
            'message': 'User updated successfully.',
            'errors': []
        })
    
    @action(detail=False, methods=['post'], permission_classes=[permissions.IsAuthenticated, CanAddRoles])
    def add_role(self, request):
        """Add a new role (manager, reviewer, etc.) - Admin only"""
        serializer = UserCreateSerializer(data=request.data, context={'request': request})
        serializer.is_valid(raise_exception=True)
        
        # Ensure the role is valid and not admin
        role = request.data.get('role', 'user')
        if role not in ['manager', 'reviewer', 'user']:
            return Response({
                'success': False,
                'data': {},
                'message': 'Invalid role specified.',
                'errors': ['Role must be manager, reviewer, or user.']
            }, status=status.HTTP_400_BAD_REQUEST)
        
        user = serializer.save()
        
        # Re-fetch with annotations for response
        try:
            user_with_counts = self.get_queryset().get(pk=user.pk)
            response_data = UserAdminSerializer(user_with_counts).data
        except User.DoesNotExist:
            response_data = UserAdminSerializer(user).data
        
        return Response({
            'success': True,
            'data': response_data,
            'message': f'{role.capitalize()} added successfully.',
            'errors': []
        }, status=status.HTTP_201_CREATED)
    
    @action(detail=False, methods=['get'])
    def user_permissions(self, request):
        """Get current user's permissions for UI display"""
        print("User Permissions Check for:", request.user)
        user = request.user
        
        
        permissions_data = {
            'can_add_roles': user.can_add_roles(),
            'can_manage_users': user.can_manage_users(),
            'role': user.role,
            'is_admin': user.is_admin(),
            'is_manager': user.is_manager(),
            'is_reviewer': user.is_reviewer()
        }
        
        return Response({
            'success': True,
            'data': permissions_data,
            'message': 'User permissions retrieved successfully.',
            'errors': []
        })
    
    @action(detail=True, methods=['post'])
    def block(self, request, pk=None):
        """Block a user."""
        user = self.get_object()
        user.block_user()
        
        # Re-fetch with annotations
        user = self.get_queryset().get(pk=user.pk)
        return Response({
            'success': True,
            'data': UserAdminSerializer(user).data,
            'message': 'User blocked successfully.',
            'errors': []
        })
    
    @action(detail=True, methods=['post'])
    def unblock(self, request, pk=None):
        """Unblock a user."""
        user = self.get_object()
        user.unblock_user()
        
        # Re-fetch with annotations
        user = self.get_queryset().get(pk=user.pk)
        return Response({
            'success': True,
            'data': UserAdminSerializer(user).data,
            'message': 'User unblocked successfully.',
            'errors': []
        })
    
    @action(detail=True, methods=['post'])
    def activate(self, request, pk=None):
        """Activate a user."""
        user = self.get_object()
        user.is_active = True
        user.save()
        
        # Re-fetch with annotations
        user = self.get_queryset().get(pk=user.pk)
        return Response({
            'success': True,
            'data': UserAdminSerializer(user).data,
            'message': 'User activated successfully.',
            'errors': []
        })
    
    @action(detail=True, methods=['post'])
    def deactivate(self, request, pk=None):
        """Deactivate a user."""
        user = self.get_object()
        user.is_active = False
        user.save()
        
        # Re-fetch with annotations
        user = self.get_queryset().get(pk=user.pk)
        return Response({
            'success': True,
            'data': UserAdminSerializer(user).data,
            'message': 'User deactivated successfully.',
            'errors': []
        })
    
    @action(detail=True, methods=['post'])
    def update_status(self, request, pk=None):
        """Update user status based on action."""
        user = self.get_object()
        action = request.data.get('action')
        
        if action == 'Block':
            user.block_user()
            message = 'User blocked successfully.'
        elif action == 'Unblock':
            user.unblock_user()
            message = 'User unblocked successfully.'
        elif action == 'Delete':
            # For delete, we might want to deactivate instead of hard delete
            user.is_active = False
            user.save()
            message = 'User deactivated successfully.'
        else:
            return Response({
                'success': False,
                'data': {},
                'message': 'Invalid action.',
                'errors': ['Action must be Block, Unblock, or Delete.']
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Re-fetch with annotations
        user = self.get_queryset().get(pk=user.pk)
        return Response({
            'success': True,
            'data': UserAdminSerializer(user).data,
            'message': message,
            'errors': []
        })

class CameraAdminViewSet(viewsets.ModelViewSet):
    """ViewSet for admin management of cameras."""
    
    permission_classes = [permissions.IsAuthenticated, IsManagerOrAdminOrReviewer]
    
    def get_queryset(self):
        """Get all cameras with user annotations."""
        queryset = Camera.objects.select_related('user').order_by('-created_at')
        
        # Apply filters if provided
        user_filter = self.request.query_params.get('user_id')
        if user_filter:
            queryset = queryset.filter(user_id=user_filter)
        
        status_filter = self.request.query_params.get('status')
        if status_filter:
            queryset = queryset.filter(status=status_filter)
        
        search = self.request.query_params.get('search')
        if search:
            queryset = queryset.filter(
                Q(name__icontains=search) |
                Q(user__full_name__icontains=search) |
                Q(user__email__icontains=search) |
                Q(stream_url__icontains=search)
            )
        
        return queryset
    
    def list(self, request, *args, **kwargs):
        """List all cameras for admin/manager."""
        queryset = self.filter_queryset(self.get_queryset())
        page = self.paginate_queryset(queryset)
        
        if page is not None:
            cameras_data = []
            for camera in page:
                cameras_data.append({
                    'id': str(camera.id),
                    'name': camera.name,
                    'stream_url': camera.stream_url,
                    'status': camera.status,
                    'user_id': str(camera.user.id),
                    'user_name': camera.user.full_name,
                    'user_email': camera.user.email,
                    'created_at': camera.created_at,
                    'updated_at': camera.updated_at,
                })
            
            response = self.get_paginated_response(cameras_data)
            response.data = {
                'success': True,
                'data': response.data,
                'message': 'Cameras retrieved successfully.',
                'errors': []
            }
            return response
        
        cameras_data = []
        for camera in queryset:
            cameras_data.append({
                'id': str(camera.id),
                'name': camera.name,
                'stream_url': camera.stream_url,
                'status': camera.status,
                'user_id': str(camera.user.id),
                'user_name': camera.user.full_name,
                'user_email': camera.user.email,
                'created_at': camera.created_at,
                'updated_at': camera.updated_at,
            })
        
        return Response({
            'success': True,
            'data': cameras_data,
            'message': 'Cameras retrieved successfully.',
            'errors': []
        })
    
    def retrieve(self, request, *args, **kwargs):
        """Retrieve a specific camera."""
        instance = self.get_object()
        
        camera_data = {
            'id': str(instance.id),
            'name': instance.name,
            'stream_url': instance.stream_url,
            'status': instance.status,
            'user_id': str(instance.user.id),
            'user_name': instance.user.full_name,
            'user_email': instance.user.email,
            'created_at': instance.created_at,
            'updated_at': instance.updated_at,
        }
        
        return Response({
            'success': True,
            'data': camera_data,
            'message': 'Camera retrieved successfully.',
            'errors': []
        })
    
    def update(self, request, *args, **kwargs):
        """Update a camera."""
        partial = kwargs.pop('partial', False)
        instance = self.get_object()
        
        # Simple update logic - you can expand this based on your Camera model fields
        allowed_fields = ['name', 'stream_url', 'status']
        update_data = {k: v for k, v in request.data.items() if k in allowed_fields}
        
        for field, value in update_data.items():
            setattr(instance, field, value)
        
        instance.save()
        
        camera_data = {
            'id': str(instance.id),
            'name': instance.name,
            'stream_url': instance.stream_url,
            'status': instance.status,
            'user_id': str(instance.user.id),
            'user_name': instance.user.full_name,
            'user_email': instance.user.email,
            'created_at': instance.created_at,
            'updated_at': instance.updated_at,
        }
        
        return Response({
            'success': True,
            'data': camera_data,
            'message': 'Camera updated successfully.',
            'errors': []
        })
    
    def destroy(self, request, *args, **kwargs):
        """Delete a camera."""
        instance = self.get_object()
        instance.delete()
        
        return Response({
            'success': True,
            'data': {},
            'message': 'Camera deleted successfully.',
            'errors': []
        })

class SystemStatusViewSet(viewsets.GenericViewSet):
    """ViewSet for system status and metrics."""
    
    permission_classes = [permissions.IsAuthenticated, IsAdminUser]
    serializer_class = SystemStatusSerializer
    
    @action(detail=False, methods=['get'])
    def status(self, request):
        """Get system status metrics."""
        try:
            # Get user counts
            total_users = User.objects.count()
            active_users = User.objects.filter(is_active=True).count()
            
            # Get camera counts
            total_cameras = Camera.objects.count()
            online_cameras = Camera.objects.filter(status='online').count()
            offline_cameras = Camera.objects.filter(status__in=['offline', 'error', 'inactive']).count()
            
            # Get alert counts
            total_alerts = Alert.objects.count()
            new_alerts = Alert.objects.filter(status='new').count()
            
            today = timezone.now().date()
            alerts_today = Alert.objects.filter(detection_time__date=today).count()
            
            # Get weekly alerts (last 7 days)
            week_start = today - datetime.timedelta(days=7)
            alerts_this_week = Alert.objects.filter(detection_time__date__gte=week_start).count()
            
            # Get alert types
            alert_types = dict(Alert.objects.values('alert_type').annotate(count=Count('id')).values_list('alert_type', 'count'))
            
            # Get system resources
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            disk = psutil.disk_usage('/')
            disk_usage = disk.percent
            
            # Determine system health based on resource usage
            if cpu_usage > 90 or memory_usage > 90 or disk_usage > 90:
                system_health = 'critical'
            elif cpu_usage > 70 or memory_usage > 70 or disk_usage > 70:
                system_health = 'warning'
            else:
                system_health = 'good'
            
            # Get system uptime
            boot_time = datetime.datetime.fromtimestamp(psutil.boot_time())
            uptime = timezone.now() - timezone.make_aware(boot_time)
            uptime_str = str(uptime).split('.')[0]  # Remove microseconds
            
            # Get last backup time (dummy value for demo)
            last_backup = timezone.now() - datetime.timedelta(days=1)
            
            # Get system version
            system_version = "1.0.0"  # Replace with actual version
            
            # Create status data
            status_data = {
                'total_users': total_users,
                'active_users': active_users,
                'total_cameras': total_cameras,
                'online_cameras': online_cameras,
                'offline_cameras': offline_cameras,
                'total_alerts': total_alerts,
                'new_alerts': new_alerts,
                'alerts_today': alerts_today,
                'alerts_this_week': alerts_this_week,
                'alert_types': alert_types,
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage,
                'disk_usage': disk_usage,
                'system_health': system_health,
                'uptime': uptime_str,
                'last_backup': last_backup,
                'system_version': system_version
            }
            
            serializer = self.get_serializer(status_data)
            
            # Create a system check record
            SystemCheck.objects.create(
                check_type='auto',
                status='success' if system_health != 'critical' else 'error',
                details=f"CPU: {cpu_usage}%, Memory: {memory_usage}%, Disk: {disk_usage}%",
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                camera_count=total_cameras,
                online_cameras=online_cameras,
                offline_cameras=offline_cameras,
                alerts_24h=alerts_today
            )
            
            return Response({
                'success': True,
                'data': serializer.data,
                'message': 'System status retrieved successfully.',
                'errors': []
            })
            
        except Exception as e:
            logger.error(f"Error retrieving system status: {str(e)}")
            
            return Response({
                'success': False,
                'data': {},
                'message': 'Error retrieving system status.',
                'errors': [str(e)]
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class SubscriptionViewSet(viewsets.ModelViewSet):
    """ViewSet for subscription management."""
    
    permission_classes = [permissions.IsAuthenticated, IsAdminUser]
    
    def get_queryset(self):
        """Get appropriate queryset based on action."""
        if self.action in ['user_subscription', 'update_user_subscription']:
            return UserSubscription.objects.all()
        return SubscriptionPlan.objects.all()
    
    def get_serializer_class(self):
        """Return appropriate serializer class based on the action."""
        if self.action in ['user_subscription', 'update_user_subscription']:
            return UserSubscriptionSerializer
        return SubscriptionPlanSerializer
    
    @action(detail=False, methods=['get'])
    def user_subscription(self, request):
        """Get subscription details for the current user."""
        try:
            subscription = UserSubscription.objects.get(user=request.user)
            serializer = UserSubscriptionSerializer(subscription)
            
            return Response({
                'success': True,
                'data': serializer.data,
                'message': 'Subscription details retrieved successfully.',
                'errors': []
            })
        except UserSubscription.DoesNotExist:
            return Response({
                'success': False,
                'data': {},
                'message': 'No subscription found for this user.',
                'errors': ['User does not have an active subscription.']
            }, status=status.HTTP_404_NOT_FOUND)
    
    @action(detail=True, methods=['put', 'patch'])
    def update_user_subscription(self, request, pk=None):
        """Update subscription for a specific user."""
        try:
            user = User.objects.get(pk=pk)
            subscription = UserSubscription.objects.get(user=user)
            
            serializer = UserSubscriptionSerializer(
                subscription, data=request.data, partial=True
            )
            serializer.is_valid(raise_exception=True)
            serializer.save()
            
            return Response({
                'success': True,
                'data': serializer.data,
                'message': 'Subscription updated successfully.',
                'errors': []
            })
        except User.DoesNotExist:
            return Response({
                'success': False,
                'data': {},
                'message': 'User not found.',
                'errors': ['Invalid user ID.']
            }, status=status.HTTP_404_NOT_FOUND)
        except UserSubscription.DoesNotExist:
            return Response({
                'success': False,
                'data': {},
                'message': 'No subscription found for this user.',
                'errors': ['User does not have an active subscription.']
            }, status=status.HTTP_404_NOT_FOUND)

class SystemSettingViewSet(viewsets.ModelViewSet):
    """ViewSet for system settings management."""
    
    permission_classes = [permissions.IsAuthenticated, IsAdminUser]
    serializer_class = SystemSettingSerializer
    queryset = SystemSetting.objects.all()
    
    def perform_create(self, serializer):
        """Save the system setting with the current user."""
        serializer.save(updated_by=self.request.user)
    
    def perform_update(self, serializer):
        """Update the system setting with the current user."""
        serializer.save(updated_by=self.request.user)
    
    @action(detail=False, methods=['get'])
    def by_category(self, request):
        """Get settings grouped by category."""
        settings = SystemSetting.objects.all()
        
        # Group settings by category
        categories = {}
        for setting in settings:
            category = setting.category or 'Uncategorized'
            if category not in categories:
                categories[category] = []
            
            categories[category].append(SystemSettingSerializer(setting).data)
        
        return Response({
            'success': True,
            'data': categories,
            'message': 'Settings retrieved successfully.',
            'errors': []
        })