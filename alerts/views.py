from rest_framework import viewsets, status, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from django.db.models import Count
from django.utils import timezone
import datetime
import os
from django.conf import settings
from django.http import FileResponse
import logging

from .models import Alert
from .serializers import (
    AlertSerializer, AlertListSerializer, AlertCreateSerializer,
    AlertStatusUpdateSerializer, AlertSummarySerializer
)
from utils.permissions import IsOwnerOrAdmin

logger = logging.getLogger('security_ai')

class AlertViewSet(viewsets.ModelViewSet):
    """ViewSet for managing alerts."""
    
    permission_classes = [permissions.IsAuthenticated, IsOwnerOrAdmin]
    serializer_class = AlertSerializer
    
    def get_queryset(self):
        """Return alerts based on user permissions."""
        user = self.request.user
        
        # For admin users, return all alerts
        if user.is_admin():
            queryset = Alert.objects.all()
        else:
            # For regular users, return alerts from their cameras
            queryset = Alert.objects.filter(camera__user=user)
        
        # Apply filters if provided
        status_filter = self.request.query_params.get('status')
        if status_filter:
            queryset = queryset.filter(status=status_filter)
        
        alert_type = self.request.query_params.get('type')
        if alert_type:
            queryset = queryset.filter(alert_type=alert_type)
        
        severity = self.request.query_params.get('severity')
        if severity:
            queryset = queryset.filter(severity=severity)
        
        camera_id = self.request.query_params.get('camera_id')
        if camera_id:
            queryset = queryset.filter(camera_id=camera_id)
        
        # Date range filters
        start_date = self.request.query_params.get('start_date')
        if start_date:
            try:
                start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
                queryset = queryset.filter(detection_time__date__gte=start_date)
            except ValueError:
                pass
        
        end_date = self.request.query_params.get('end_date')
        if end_date:
            try:
                end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()
                queryset = queryset.filter(detection_time__date__lte=end_date)
            except ValueError:
                pass
        
        return queryset
    
    def get_serializer_class(self):
        """Return appropriate serializer class based on the action."""
        if self.action == 'list':
            return AlertListSerializer
        elif self.action == 'create':
            return AlertCreateSerializer
        elif self.action == 'update_status':
            return AlertStatusUpdateSerializer
        return self.serializer_class
    
    def list(self, request, *args, **kwargs):
        """List alerts with pagination."""
        queryset = self.filter_queryset(self.get_queryset())
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
    
    def retrieve(self, request, *args, **kwargs):
        """Retrieve a specific alert."""
        instance = self.get_object()
        serializer = AlertSerializer(instance)
        
        return Response({
            'success': True,
            'data': serializer.data,
            'message': 'Alert retrieved successfully.',
            'errors': []
        })
    
    def create(self, request, *args, **kwargs):
        """Create a new alert."""
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        alert = serializer.save()
        
        headers = self.get_success_headers(serializer.data)
        return Response({
            'success': True,
            'data': AlertSerializer(alert).data,
            'message': 'Alert created successfully.',
            'errors': []
        }, status=status.HTTP_201_CREATED, headers=headers)
    
    @action(detail=True, methods=['put', 'patch'])
    def status(self, request, pk=None):
        """Update the status of an alert."""
        alert = self.get_object()
        serializer = AlertStatusUpdateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        new_status = serializer.validated_data['status']
        notes = serializer.validated_data.get('notes')
        
        if notes:
            alert.add_notes(notes)
        
        if new_status == 'confirmed':
            alert.mark_as_confirmed(request.user)
        elif new_status == 'dismissed':
            alert.mark_as_dismissed(request.user)
        elif new_status == 'false_positive':
            alert.mark_as_false_positive(request.user)
        
        return Response({
            'success': True,
            'data': AlertSerializer(alert).data,
            'message': f'Alert status updated to {new_status}.',
            'errors': []
        })
    
    @action(detail=False, methods=['get'])
    def summary(self, request):
        """Get alert summary statistics."""
        queryset = self.get_queryset()
        
        # Get total counts
        total_alerts = queryset.count()
        new_alerts = queryset.filter(status='new').count()
        confirmed_alerts = queryset.filter(status='confirmed').count()
        dismissed_alerts = queryset.filter(status='dismissed').count()
        false_positive_alerts = queryset.filter(status='false_positive').count()
        
        # Get counts by type
        by_type = dict(queryset.values('alert_type').annotate(count=Count('id')).values_list('alert_type', 'count'))
        
        # Get counts by severity
        by_severity = dict(queryset.values('severity').annotate(count=Count('id')).values_list('severity', 'count'))
        
        # Get daily counts for the last 7 days
        today = timezone.now().date()
        daily_count = []
        
        for i in range(6, -1, -1):
            date = today - datetime.timedelta(days=i)
            count = queryset.filter(detection_time__date=date).count()
            daily_count.append({
                'date': date.strftime('%Y-%m-%d'),
                'count': count
            })
        
        # Get weekly counts for the last 4 weeks
        weekly_count = []
        
        for i in range(3, -1, -1):
            start_date = today - datetime.timedelta(days=(i+1)*7)
            end_date = today - datetime.timedelta(days=i*7)
            count = queryset.filter(detection_time__date__gt=start_date, detection_time__date__lte=end_date).count()
            weekly_count.append({
                'week': f"Week {4-i}",
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'count': count
            })
        
        # Get monthly counts for the last 6 months
        monthly_count = []
        
        for i in range(5, -1, -1):
            # Get the first day of the month
            first_day = (today.replace(day=1) - datetime.timedelta(days=1)).replace(day=1)
            first_day = first_day.replace(month=((today.month - i - 1) % 12) + 1)
            if first_day.month > today.month:
                first_day = first_day.replace(year=today.year - 1)
            
            # Get the last day of the month
            if i > 0:
                last_day = (first_day.replace(month=first_day.month % 12 + 1, day=1) - datetime.timedelta(days=1))
            else:
                last_day = today
            
            count = queryset.filter(detection_time__date__gte=first_day, detection_time__date__lte=last_day).count()
            monthly_count.append({
                'month': first_day.strftime('%B %Y'),
                'start_date': first_day.strftime('%Y-%m-%d'),
                'end_date': last_day.strftime('%Y-%m-%d'),
                'count': count
            })
        
        summary_data = {
            'total_alerts': total_alerts,
            'new_alerts': new_alerts,
            'confirmed_alerts': confirmed_alerts,
            'dismissed_alerts': dismissed_alerts,
            'false_positive_alerts': false_positive_alerts,
            'by_type': by_type,
            'by_severity': by_severity,
            'daily_count': daily_count,
            'weekly_count': weekly_count,
            'monthly_count': monthly_count
        }
        
        serializer = AlertSummarySerializer(summary_data)
        
        return Response({
            'success': True,
            'data': serializer.data,
            'message': 'Alert summary retrieved successfully.',
            'errors': []
        })
    
    @action(detail=True, methods=['get'])
    def video(self, request, pk=None):
        """Get video file for a specific alert."""
        alert = self.get_object()
        
        if not alert.video_file:
            return Response({
                'success': False,
                'data': {},
                'message': 'No video file available for this alert.',
                'errors': ['Video file not found.']
            }, status=status.HTTP_404_NOT_FOUND)
        
        try:
            # Get the file path
            file_path = os.path.join(settings.MEDIA_ROOT, str(alert.video_file))
            
            # Check if file exists
            if not os.path.exists(file_path):
                return Response({
                    'success': False,
                    'data': {},
                    'message': 'Video file not found on the server.',
                    'errors': ['File does not exist.']
                }, status=status.HTTP_404_NOT_FOUND)
            
            # Return the file
            response = FileResponse(open(file_path, 'rb'))
            response['Content-Disposition'] = f'inline; filename="{os.path.basename(file_path)}"'
            return response
            
        except Exception as e:
            logger.error(f"Error retrieving video file for alert {alert.id}: {str(e)}")
            return Response({
                'success': False,
                'data': {},
                'message': 'Error retrieving video file.',
                'errors': [str(e)]
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def destroy(self, request, pk=None):
        """Delete a single alert and all its related data including files."""
        from notifications.models import NotificationLog
        from django.db import transaction
        
        try:
            alert = self.get_object()
            alert_id = alert.id
            
            with transaction.atomic():
                files_deleted = {'videos': 0, 'thumbnails': 0}
                file_errors = []
                
                # Delete related NotificationLog entries
                notification_logs_deleted = NotificationLog.objects.filter(alert=alert).count()
                NotificationLog.objects.filter(alert=alert).delete()
                
                # Delete video file if exists
                if alert.video_file:
                    try:
                        video_path = os.path.join(settings.MEDIA_ROOT, str(alert.video_file))
                        if os.path.exists(video_path):
                            os.remove(video_path)
                            files_deleted['videos'] += 1
                            logger.info(f"Deleted video file: {video_path}")
                    except Exception as e:
                        file_errors.append(f"Failed to delete video: {str(e)}")
                        logger.error(f"Failed to delete video file {video_path}: {str(e)}")
                
                # Delete thumbnail file if exists
                if alert.thumbnail:
                    try:
                        thumbnail_path = os.path.join(settings.MEDIA_ROOT, str(alert.thumbnail))
                        if os.path.exists(thumbnail_path):
                            os.remove(thumbnail_path)
                            files_deleted['thumbnails'] += 1
                            logger.info(f"Deleted thumbnail file: {thumbnail_path}")
                    except Exception as e:
                        file_errors.append(f"Failed to delete thumbnail: {str(e)}")
                        logger.error(f"Failed to delete thumbnail file {thumbnail_path}: {str(e)}")
                
                # Delete the alert (this will cascade to AlertReview due to foreign key)
                alert.delete()
                
                logger.info(f"Deleted alert {alert_id} and {notification_logs_deleted} related notification logs")
                
                response_data = {
                    'success': True,
                    'data': {
                        'deleted_alert_id': alert_id,
                        'files_deleted': files_deleted
                    },
                    'message': 'Alert and all related data deleted successfully.',
                    'errors': file_errors if file_errors else []
                }
                
                if file_errors:
                    response_data['message'] += f' However, {len(file_errors)} file(s) could not be deleted.'
                
                return Response(response_data, status=status.HTTP_200_OK)
                
        except Exception as e:
            logger.error(f"Error deleting alert {pk}: {str(e)}")
            return Response({
                'success': False,
                'data': {},
                'message': 'Error occurred while deleting alert.',
                'errors': [str(e)]
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    @action(detail=False, methods=['post'], url_path='delete-multiple')
    def delete_multiple(self, request):
        """Delete multiple alerts and their related data."""
        from notifications.models import NotificationLog
        from django.db import transaction
        
        alert_ids = request.data.get('alert_ids', [])
        
        # Validate input
        if not alert_ids:
            return Response({
                'success': False,
                'data': {},
                'message': 'No alert IDs provided.',
                'errors': ['alert_ids field is required and cannot be empty.']
            }, status=status.HTTP_400_BAD_REQUEST)
        
        if not isinstance(alert_ids, list):
            return Response({
                'success': False,
                'data': {},
                'message': 'alert_ids must be an array.',
                'errors': ['alert_ids must be a list of integers.']
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Validate that all IDs are integers
        try:
            alert_ids = [int(alert_id) for alert_id in alert_ids]
        except (ValueError, TypeError):
            return Response({
                'success': False,
                'data': {},
                'message': 'Invalid alert ID format.',
                'errors': ['All alert IDs must be valid integers.']
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Get queryset based on user permissions
        base_queryset = self.get_queryset()
        alerts_to_delete = base_queryset.filter(id__in=alert_ids)
        
        # Check if all requested alerts exist and user has permission to delete them
        found_ids = list(alerts_to_delete.values_list('id', flat=True))
        not_found_ids = [aid for aid in alert_ids if aid not in found_ids]
        
        if not_found_ids:
            return Response({
                'success': False,
                'data': {'not_found_ids': not_found_ids},
                'message': f'Some alerts not found or you do not have permission to delete them.',
                'errors': [f'Alert IDs not found or no permission: {not_found_ids}']
            }, status=status.HTTP_404_NOT_FOUND)
        
        try:
            with transaction.atomic():
                deleted_count = 0
                files_deleted = {'videos': 0, 'thumbnails': 0}
                file_errors = []
                
                for alert in alerts_to_delete:
                    alert_id = alert.id
                    
                    # Delete related NotificationLog entries
                    notification_logs_deleted = NotificationLog.objects.filter(alert=alert).count()
                    NotificationLog.objects.filter(alert=alert).delete()
                    
                    # Delete video file if exists
                    if alert.video_file:
                        try:
                            video_path = os.path.join(settings.MEDIA_ROOT, str(alert.video_file))
                            if os.path.exists(video_path):
                                os.remove(video_path)
                                files_deleted['videos'] += 1
                                logger.info(f"Deleted video file: {video_path}")
                        except Exception as e:
                            file_errors.append(f"Failed to delete video for alert {alert_id}: {str(e)}")
                            logger.error(f"Failed to delete video file {video_path}: {str(e)}")
                    
                    # Delete thumbnail file if exists
                    if alert.thumbnail:
                        try:
                            thumbnail_path = os.path.join(settings.MEDIA_ROOT, str(alert.thumbnail))
                            if os.path.exists(thumbnail_path):
                                os.remove(thumbnail_path)
                                files_deleted['thumbnails'] += 1
                                logger.info(f"Deleted thumbnail file: {thumbnail_path}")
                        except Exception as e:
                            file_errors.append(f"Failed to delete thumbnail for alert {alert_id}: {str(e)}")
                            logger.error(f"Failed to delete thumbnail file {thumbnail_path}: {str(e)}")
                    
                    # Delete related AlertReview entries (will be handled by CASCADE)
                    # The Alert model has review_history with CASCADE delete
                    
                    # Delete the alert (this will cascade to AlertReview due to foreign key)
                    alert.delete()
                    deleted_count += 1
                    
                    logger.info(f"Deleted alert {alert_id} and {notification_logs_deleted} related notification logs")
                
                response_data = {
                    'success': True,
                    'data': {
                        'deleted_count': deleted_count,
                        'deleted_alert_ids': found_ids,
                        'files_deleted': files_deleted
                    },
                    'message': f'Successfully deleted {deleted_count} alerts and their related data.',
                    'errors': file_errors if file_errors else []
                }
                
                if file_errors:
                    response_data['message'] += f' However, {len(file_errors)} file(s) could not be deleted.'
                
                return Response(response_data, status=status.HTTP_200_OK)
                
        except Exception as e:
            logger.error(f"Error deleting alerts {alert_ids}: {str(e)}")
            return Response({
                'success': False,
                'data': {},
                'message': 'Error occurred while deleting alerts.',
                'errors': [str(e)]
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)