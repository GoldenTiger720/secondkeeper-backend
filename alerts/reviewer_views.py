# alerts/reviewer_views.py - API views for reviewer workflow

from rest_framework import viewsets, status, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from django.db.models import Count, Q
from django.utils import timezone
from django.contrib.auth import get_user_model
import logging
import os
import shutil
from django.conf import settings

from .models import Alert, AlertReview, ReviewerAssignment
from .serializers import (
    AlertSerializer, AlertReviewSerializer, ReviewerAssignmentSerializer,
    AlertStatusUpdateSerializer
)
from utils.permissions import IsReviewerOrAbove, IsAdminUser
from training.models import TrainingFire, TrainingChoking, TrainingFall, TrainingViolence

User = get_user_model()
logger = logging.getLogger('security_ai')

def move_image_to_datasets(alert):
    """
    Move thumbnail image to appropriate Datasets folder based on alert type.
    Returns the new file path if successful, None if failed.
    """
    if not alert.thumbnail or not alert.thumbnail.name:
        logger.warning(f"No thumbnail found for alert {alert.id}")
        return None
    
    try:
        # Get the current thumbnail path
        current_path = alert.thumbnail.path
        
        # Check if the file exists
        if not os.path.exists(current_path):
            logger.warning(f"Thumbnail file not found at {current_path} for alert {alert.id}")
            return None
        
        # Define dataset paths based on alert type
        datasets_mapping = {
            'fire_smoke': 'fire/NonFire',
            'fall': 'fall/NonFall', 
            'choking': 'choking/NonChoking',
            'violence': 'violence/NonViolence'
        }
        
        if alert.alert_type not in datasets_mapping:
            logger.warning(f"Alert type {alert.alert_type} not supported for dataset training")
            return None
        
        # Build destination path
        datasets_base = os.path.join(settings.MEDIA_ROOT, 'Datasets')
        destination_dir = os.path.join(datasets_base, datasets_mapping[alert.alert_type])
        
        # Create destination directory if it doesn't exist
        os.makedirs(destination_dir, exist_ok=True)
        
        # Get filename from original path
        filename = os.path.basename(current_path)
        destination_path = os.path.join(destination_dir, filename)
        
        # Move the file (cut, not copy)
        shutil.copy2(current_path, destination_path)
        
        # Return the relative path for database storage
        relative_path = os.path.relpath(destination_path, settings.MEDIA_ROOT)
        logger.info(f"Moved thumbnail for alert {alert.id} to {relative_path}")
        
        return relative_path
        
    except Exception as e:
        logger.error(f"Error moving thumbnail for alert {alert.id}: {str(e)}")
        return None

def create_training_record(alert, moved_file_path):
    """
    Create a training record in the appropriate training table.
    """
    try:
        # Define image types for each alert type
        image_type_mapping = {
            'fire_smoke': 'NonFire',
            'fall': 'NonFall',
            'choking': 'NonChoking', 
            'violence': 'NonViolence'
        }
        
        # Define model mapping
        model_mapping = {
            'fire_smoke': TrainingFire,
            'fall': TrainingFall,
            'choking': TrainingChoking,
            'violence': TrainingViolence
        }
        
        if alert.alert_type not in model_mapping:
            logger.warning(f"Alert type {alert.alert_type} not supported for training record")
            return False
        
        # Get the appropriate model and image type
        model_class = model_mapping[alert.alert_type]
        image_type = image_type_mapping[alert.alert_type]
        
        # Create the training record
        training_record = model_class.objects.create(
            image_type=image_type,
            image_url=moved_file_path
        )
        
        logger.info(f"Created training record {training_record.id} for alert {alert.id}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating training record for alert {alert.id}: {str(e)}")
        return False

def copy_image_to_positive_datasets(alert):
    """
    Copy thumbnail image to appropriate positive Datasets folder based on alert type.
    Returns the new file path if successful, None if failed.
    """
    if not alert.thumbnail or not alert.thumbnail.name:
        logger.warning(f"No thumbnail found for alert {alert.id}")
        return None
    
    try:
        # Get the current thumbnail path
        current_path = alert.thumbnail.path
        
        # Check if the file exists
        if not os.path.exists(current_path):
            logger.warning(f"Thumbnail file not found at {current_path} for alert {alert.id}")
            return None
        
        # Define dataset paths based on alert type (positive folders)
        datasets_mapping = {
            'fire_smoke': 'fire/Fire',
            'fall': 'fall/Fall', 
            'choking': 'choking/Choking',
            'violence': 'violence/Violence'
        }
        
        if alert.alert_type not in datasets_mapping:
            logger.warning(f"Alert type {alert.alert_type} not supported for positive dataset training")
            return None
        
        # Build destination path
        datasets_base = os.path.join(settings.MEDIA_ROOT, 'Datasets')
        destination_dir = os.path.join(datasets_base, datasets_mapping[alert.alert_type])
        
        # Create destination directory if it doesn't exist
        os.makedirs(destination_dir, exist_ok=True)
        
        # Get filename from original path
        filename = os.path.basename(current_path)
        destination_path = os.path.join(destination_dir, filename)
        
        # Copy the file (not move, since we want to keep the original)
        shutil.copy2(current_path, destination_path)
        
        # Return the relative path for database storage
        relative_path = os.path.relpath(destination_path, settings.MEDIA_ROOT)
        logger.info(f"Copied thumbnail for alert {alert.id} to {relative_path}")
        
        return relative_path
        
    except Exception as e:
        logger.error(f"Error copying thumbnail for alert {alert.id}: {str(e)}")
        return None

def create_positive_training_record(alert, copied_file_path):
    """
    Create a positive training record in the appropriate training table.
    """
    try:
        # Define image types for each alert type (positive classes)
        image_type_mapping = {
            'fire_smoke': 'Fire',
            'fall': 'Fall',
            'choking': 'Choking', 
            'violence': 'Violence'
        }
        
        # Define model mapping
        model_mapping = {
            'fire_smoke': TrainingFire,
            'fall': TrainingFall,
            'choking': TrainingChoking,
            'violence': TrainingViolence
        }
        
        if alert.alert_type not in model_mapping:
            logger.warning(f"Alert type {alert.alert_type} not supported for positive training record")
            return False
        
        # Get the appropriate model and image type
        model_class = model_mapping[alert.alert_type]
        image_type = image_type_mapping[alert.alert_type]
        
        # Create the training record
        training_record = model_class.objects.create(
            image_type=image_type,
            image_url=copied_file_path
        )
        
        logger.info(f"Created positive training record {training_record.id} for alert {alert.id}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating positive training record for alert {alert.id}: {str(e)}")
        return False

class ReviewerAlertViewSet(viewsets.ModelViewSet):
    """ViewSet for reviewers to manage pending alerts."""
    
    permission_classes = [permissions.IsAuthenticated, IsReviewerOrAbove]
    serializer_class = AlertSerializer
    
    def get_queryset(self):
        """Return alerts based on reviewer permissions."""
        user = self.request.user
        
        if user.is_admin():
            # Admins can see all alerts
            return Alert.objects.all().select_related('camera', 'camera__user').order_by('-detection_time')
        elif user.is_reviewer():
            # Reviewers see alerts assigned to them or pending alerts
            return Alert.objects.filter(
                Q(status='pending_review') | Q(reviewed_by=user)
            ).select_related('camera', 'camera__user').order_by('-detection_time')
        else:
            # Regular users shouldn't access this viewset
            return Alert.objects.none()
    
    def list(self, request, *args, **kwargs):
        """List alerts for review."""
        queryset = self.filter_queryset(self.get_queryset())
        
        # Apply filters
        status_filter = request.query_params.get('status')
        if status_filter:
            queryset = queryset.filter(status=status_filter)
        
        alert_type = request.query_params.get('type')
        if alert_type:
            queryset = queryset.filter(alert_type=alert_type)
        
        severity = request.query_params.get('severity')
        if severity:
            queryset = queryset.filter(severity=severity)
        
        # Filter for pending alerts by default
        if not status_filter:
            queryset = queryset.filter(status='pending_review')
        
        page = self.paginate_queryset(queryset)
        if page is not None:
            serializer = self.get_serializer(page, many=True)
            response = self.get_paginated_response(serializer.data)
            response.data = {
                'success': True,
                'data': response.data,
                'message': 'Alerts retrieved successfully.',
                'errors': []
            }
            return response
        
        serializer = self.get_serializer(queryset, many=True)
        return Response({
            'success': True,
            'data': serializer.data,
            'message': 'Alerts retrieved successfully.',
            'errors': []
        })
    
    @action(detail=True, methods=['post'])
    def confirm(self, request, pk=None):
        """Confirm an alert as true positive and copy image to positive training dataset."""
        try:
            alert = self.get_object()
            
            if alert.status != 'pending_review':
                return Response({
                    'success': False,
                    'data': {},
                    'message': 'Alert is not pending review.',
                    'errors': ['Invalid alert status.']
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Get reviewer notes if provided
            notes = request.data.get('notes', '')
            if notes:
                alert.reviewer_notes = notes
            
            # Copy thumbnail image to positive Datasets folder
            copied_file_path = copy_image_to_positive_datasets(alert)
            
            training_record_created = False
            if copied_file_path:
                # Create positive training record in appropriate table
                training_record_created = create_positive_training_record(alert, copied_file_path)
            
            # Mark as confirmed and send notification to user
            alert.mark_as_confirmed(request.user, send_notification=True)
            
            # Create review record
            AlertReview.objects.create(
                alert=alert,
                reviewer=request.user,
                action='confirmed',
                notes=notes
            )
            
            logger.info(f"Alert {alert.id} confirmed by reviewer {request.user.id}")
            
            response_message = 'Alert confirmed and user notified.'
            if copied_file_path and training_record_created:
                response_message += ' Image copied to positive training dataset.'
            elif copied_file_path:
                response_message += ' Image copied but training record creation failed.'
            elif alert.thumbnail:
                response_message += ' Failed to copy image to training dataset.'
            
            return Response({
                'success': True,
                'data': AlertSerializer(alert).data,
                'message': response_message,
                'errors': []
            })
            
        except Exception as e:
            logger.error(f"Error confirming alert: {str(e)}")
            return Response({
                'success': False,
                'data': {},
                'message': 'Error confirming alert.',
                'errors': [str(e)]
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    @action(detail=True, methods=['post'])
    def dismiss(self, request, pk=None):
        """Dismiss an alert as not actionable."""
        try:
            alert = self.get_object()
            
            if alert.status != 'pending_review':
                return Response({
                    'success': False,
                    'data': {},
                    'message': 'Alert is not pending review.',
                    'errors': ['Invalid alert status.']
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Get reviewer notes
            notes = request.data.get('notes', '')
            
            # Mark as dismissed
            alert.mark_as_dismissed(request.user, notes)
            
            # Create review record
            AlertReview.objects.create(
                alert=alert,
                reviewer=request.user,
                action='dismissed',
                notes=notes
            )
            
            logger.info(f"Alert {alert.id} dismissed by reviewer {request.user.id}")
            
            return Response({
                'success': True,
                'data': AlertSerializer(alert).data,
                'message': 'Alert dismissed.',
                'errors': []
            })
            
        except Exception as e:
            logger.error(f"Error dismissing alert: {str(e)}")
            return Response({
                'success': False,
                'data': {},
                'message': 'Error dismissing alert.',
                'errors': [str(e)]
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    @action(detail=True, methods=['post'])
    def mark_false_positive(self, request, pk=None):
        """Mark an alert as false positive."""
        try:
            alert = self.get_object()
            
            if alert.status != 'pending_review':
                return Response({
                    'success': False,
                    'data': {},
                    'message': 'Alert is not pending review.',
                    'errors': ['Invalid alert status.']
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Get reviewer notes
            notes = request.data.get('notes', '')
            
            # Mark as false positive
            alert.mark_as_false_positive(request.user, notes)
            
            # Create review record
            AlertReview.objects.create(
                alert=alert,
                reviewer=request.user,
                action='false_positive',
                notes=notes
            )
            
            logger.info(f"Alert {alert.id} marked as false positive by reviewer {request.user.id}")
            
            return Response({
                'success': True,
                'data': AlertSerializer(alert).data,
                'message': 'Alert marked as false positive.',
                'errors': []
            })
            
        except Exception as e:
            logger.error(f"Error marking alert as false positive: {str(e)}")
            return Response({
                'success': False,
                'data': {},
                'message': 'Error marking alert as false positive.',
                'errors': [str(e)]
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    @action(detail=True, methods=['post'], url_path='false-positive')
    def false_positive(self, request, pk=None):
        """Mark an alert as false positive and move image to training dataset."""
        try:
            alert = self.get_object()
            
            if alert.status != 'pending_review':
                return Response({
                    'success': False,
                    'data': {},
                    'message': 'Alert is not pending review.',
                    'errors': ['Invalid alert status.']
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Get reviewer notes
            notes = request.data.get('notes', '')
            
            # Move thumbnail image to Datasets folder
            moved_file_path = move_image_to_datasets(alert)
            
            training_record_created = False
            if moved_file_path:
                # Create training record in appropriate table
                training_record_created = create_training_record(alert, moved_file_path)
            
            # Mark as false positive
            alert.mark_as_false_positive(request.user, notes)
            
            # Create review record
            AlertReview.objects.create(
                alert=alert,
                reviewer=request.user,
                action='false_positive',
                notes=notes
            )
            
            logger.info(f"Alert {alert.id} marked as false positive by reviewer {request.user.id}")
            
            response_message = 'Alert marked as false positive.'
            if moved_file_path and training_record_created:
                response_message += ' Image moved to training dataset.'
            elif moved_file_path:
                response_message += ' Image moved but training record creation failed.'
            elif alert.thumbnail:
                response_message += ' Failed to move image to training dataset.'
            
            return Response({
                'success': True,
                'data': AlertSerializer(alert).data,
                'message': response_message,
                'errors': []
            })
            
        except Exception as e:
            logger.error(f"Error marking alert as false positive: {str(e)}")
            return Response({
                'success': False,
                'data': {},
                'message': 'Error marking alert as false positive.',
                'errors': [str(e)]
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    
    @action(detail=False, methods=['get'])
    def pending_summary(self, request):
        """Get summary of pending alerts for reviewer dashboard."""
        try:
            user = request.user
            
            # Get base queryset based on user role
            if user.is_admin():
                base_queryset = Alert.objects.all()
            else:
                base_queryset = Alert.objects.filter(status='pending_review')
            
            # Count alerts by type
            pending_by_type = base_queryset.filter(status='pending_review').values(
                'alert_type'
            ).annotate(count=Count('id'))
            
            # Count alerts by severity
            pending_by_severity = base_queryset.filter(status='pending_review').values(
                'severity'
            ).annotate(count=Count('id'))
            
            # Count total pending
            total_pending = base_queryset.filter(status='pending_review').count()
            
            # Count urgent alerts (high/critical severity)
            urgent_alerts = base_queryset.filter(
                status='pending_review',
                severity__in=['high', 'critical']
            ).count()
            
            # Get oldest pending alert
            oldest_pending = base_queryset.filter(status='pending_review').order_by('detection_time').first()
            oldest_pending_hours = None
            if oldest_pending:
                time_diff = timezone.now() - oldest_pending.detection_time
                oldest_pending_hours = round(time_diff.total_seconds() / 3600, 1)
            
            summary_data = {
                'total_pending': total_pending,
                'urgent_alerts': urgent_alerts,
                'oldest_pending_hours': oldest_pending_hours,
                'by_type': {item['alert_type']: item['count'] for item in pending_by_type},
                'by_severity': {item['severity']: item['count'] for item in pending_by_severity},
                'reviewer_stats': {
                    'name': user.full_name,
                    'role': user.role,
                    'can_confirm': user.role in ['reviewer', 'admin'],
                }
            }
            
            return Response({
                'success': True,
                'data': summary_data,
                'message': 'Pending alerts summary retrieved successfully.',
                'errors': []
            })
            
        except Exception as e:
            logger.error(f"Error getting pending summary: {str(e)}")
            return Response({
                'success': False,
                'data': {},
                'message': 'Error retrieving pending summary.',
                'errors': [str(e)]
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ReviewerAllAlertsViewSet(viewsets.ReadOnlyModelViewSet):
    """ViewSet for getting all alerts (for /api/alerts/reviewer/all endpoint)."""
    
    permission_classes = [permissions.IsAuthenticated, IsReviewerOrAbove]
    serializer_class = AlertSerializer
    pagination_class = None  # Disable pagination to return all alerts
    
    def get_queryset(self):
        """Return all alerts for reviewers/admins."""
        return Alert.objects.all().select_related('camera', 'camera__user', 'resolved_by', 'reviewed_by').order_by('-detection_time')
    
    def list(self, request, *args, **kwargs):
        """Get all alerts from the database for reviewers."""
        try:
            user = request.user
            
            # Only allow reviewers and admins to access all alerts
            if not (user.is_reviewer() or user.is_admin()):
                return Response({
                    'success': False,
                    'data': {},
                    'message': 'Access denied. Reviewer or admin role required.',
                    'errors': ['Insufficient permissions.']
                }, status=status.HTTP_403_FORBIDDEN)
            
            # Get all alerts from database
            queryset = self.filter_queryset(self.get_queryset())
            
            # Apply optional filters
            alert_type = request.query_params.get('type')
            if alert_type:
                queryset = queryset.filter(alert_type=alert_type)
            
            severity = request.query_params.get('severity')
            if severity:
                queryset = queryset.filter(severity=severity)
            
            status_filter = request.query_params.get('status')
            if status_filter:
                queryset = queryset.filter(status=status_filter)
            
            camera_id = request.query_params.get('camera_id')
            if camera_id:
                queryset = queryset.filter(camera_id=camera_id)
            
            # Return all results (pagination disabled)
            serializer = self.get_serializer(queryset, many=True)
            return Response({
                'success': True,
                'data': serializer.data,
                'message': 'All alerts retrieved successfully.',
                'errors': []
            })
            
        except Exception as e:
            logger.error(f"Error getting all alerts: {str(e)}")
            return Response({
                'success': False,
                'data': {},
                'message': 'Error retrieving all alerts.',
                'errors': [str(e)]
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class AlertReviewViewSet(viewsets.ModelViewSet):
    """ViewSet for managing alert reviews."""
    
    permission_classes = [permissions.IsAuthenticated, IsReviewerOrAbove]
    serializer_class = AlertReviewSerializer
    
    def get_queryset(self):
        """Return review history based on user permissions."""
        user = self.request.user
        
        if user.is_admin():
            return AlertReview.objects.all().select_related('alert', 'reviewer')
        else:
            return AlertReview.objects.filter(reviewer=user).select_related('alert', 'reviewer')
    
    def list(self, request, *args, **kwargs):
        """List alert reviews."""
        queryset = self.filter_queryset(self.get_queryset())
        
        # Apply filters
        alert_id = request.query_params.get('alert_id')
        if alert_id:
            queryset = queryset.filter(alert_id=alert_id)
        
        action_filter = request.query_params.get('action')
        if action_filter:
            queryset = queryset.filter(action=action_filter)
        
        page = self.paginate_queryset(queryset)
        if page is not None:
            serializer = self.get_serializer(page, many=True)
            response = self.get_paginated_response(serializer.data)
            response.data = {
                'success': True,
                'data': response.data,
                'message': 'Alert reviews retrieved successfully.',
                'errors': []
            }
            return response
        
        serializer = self.get_serializer(queryset, many=True)
        return Response({
            'success': True,
            'data': serializer.data,
            'message': 'Alert reviews retrieved successfully.',
            'errors': []
        })


class ReviewerAssignmentViewSet(viewsets.ModelViewSet):
    """ViewSet for managing reviewer assignments (Admin only)."""
    
    permission_classes = [permissions.IsAuthenticated, IsAdminUser]
    serializer_class = ReviewerAssignmentSerializer
    queryset = ReviewerAssignment.objects.all().select_related('reviewer')
    
    def list(self, request, *args, **kwargs):
        """List reviewer assignments."""
        queryset = self.filter_queryset(self.get_queryset())
        
        # Apply filters
        reviewer_id = request.query_params.get('reviewer_id')
        if reviewer_id:
            queryset = queryset.filter(reviewer_id=reviewer_id)
        
        is_active = request.query_params.get('is_active')
        if is_active is not None:
            queryset = queryset.filter(is_active=is_active.lower() == 'true')
        
        serializer = self.get_serializer(queryset, many=True)
        return Response({
            'success': True,
            'data': serializer.data,
            'message': 'Reviewer assignments retrieved successfully.',
            'errors': []
        })
    
    def create(self, request, *args, **kwargs):
        """Create a new reviewer assignment."""
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        assignment = serializer.save()
        
        return Response({
            'success': True,
            'data': serializer.data,
            'message': 'Reviewer assignment created successfully.',
            'errors': []
        }, status=status.HTTP_201_CREATED)
    
    @action(detail=False, methods=['get'])
    def available_reviewers(self, request):
        """Get list of users who can be assigned as reviewers."""
        try:
            reviewers = User.objects.filter(
                role__in=['reviewer', 'admin'],
                is_active=True,
                status='active'
            ).values('id', 'full_name', 'email', 'role')
            
            return Response({
                'success': True,
                'data': list(reviewers),
                'message': 'Available reviewers retrieved successfully.',
                'errors': []
            })
            
        except Exception as e:
            logger.error(f"Error getting available reviewers: {str(e)}")
            return Response({
                'success': False,
                'data': {},
                'message': 'Error retrieving available reviewers.',
                'errors': [str(e)]
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    @action(detail=False, methods=['get'])
    def reviewer_workload(self, request):
        """Get reviewer workload statistics."""
        try:
            # Get pending alerts count per reviewer
            reviewer_workload = []
            
            reviewers = User.objects.filter(
                role__in=['reviewer', 'admin'],
                is_active=True,
                status='active'
            )
            
            for reviewer in reviewers:
                pending_count = Alert.objects.filter(
                    status='pending_review'
                ).count() if reviewer.is_admin() else Alert.objects.filter(
                    status='pending_review'
                ).count()  # For now, all reviewers see all pending alerts
                
                resolved_today = AlertReview.objects.filter(
                    reviewer=reviewer,
                    review_time__date=timezone.now().date()
                ).count()
                
                reviewer_workload.append({
                    'reviewer_id': reviewer.id,
                    'reviewer_name': reviewer.full_name,
                    'pending_alerts': pending_count,
                    'resolved_today': resolved_today,
                    'role': reviewer.role
                })
            
            return Response({
                'success': True,
                'data': reviewer_workload,
                'message': 'Reviewer workload retrieved successfully.',
                'errors': []
            })
            
        except Exception as e:
            logger.error(f"Error getting reviewer workload: {str(e)}")
            return Response({
                'success': False,
                'data': {},
                'message': 'Error retrieving reviewer workload.',
                'errors': [str(e)]
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)