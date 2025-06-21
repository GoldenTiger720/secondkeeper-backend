# alerts/reviewer_approval_views.py - New system for approving detections

from rest_framework import viewsets, status, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from django.db.models import Count, Q
from django.utils import timezone
from django.contrib.auth import get_user_model
from django.db import transaction
import logging
import json
import os
from django.conf import settings

from .models import Alert, AlertReview, ReviewerAssignment
from .serializers import AlertSerializer, AlertReviewSerializer
from utils.permissions import IsReviewerOrAbove, IsAdminUser
from utils.camera_detection_manager import detection_manager

User = get_user_model()
logger = logging.getLogger('security_ai')

class ReviewerApprovalViewSet(viewsets.GenericViewSet):
    """ViewSet for reviewing and approving pending detections"""
    
    permission_classes = [permissions.IsAuthenticated, IsReviewerOrAbove]
    
    @action(detail=False, methods=['get'])
    def pending_detections(self, request):
        """Get list of pending detections from filesystem"""
        try:
            user = request.user
            
            # Apply filters
            user_id_filter = request.query_params.get('user_id')
            alert_type_filter = request.query_params.get('alert_type')
            
            # Get pending detections
            pending_detections = detection_manager.get_pending_detections(
                user_id=user_id_filter,
                alert_type=alert_type_filter
            )
            
            # Add additional info for each detection
            enriched_detections = []
            for detection in pending_detections:
                # Get relative video path for web access
                video_path = detection.get('video_path', '')
                if video_path and video_path.startswith(settings.MEDIA_ROOT):
                    detection['video_url'] = video_path.replace(settings.MEDIA_ROOT, '/media').replace('\\', '/')
                
                # Get relative thumbnail path
                thumbnail_path = detection.get('thumbnail_path', '')
                if thumbnail_path and thumbnail_path.startswith(settings.MEDIA_ROOT):
                    detection['thumbnail_url'] = thumbnail_path.replace(settings.MEDIA_ROOT, '/media').replace('\\', '/')
                
                enriched_detections.append(detection)
            
            return Response({
                'success': True,
                'data': {
                    'detections': enriched_detections,
                    'total_count': len(enriched_detections)
                },
                'message': 'Pending detections retrieved successfully.',
                'errors': []
            })
            
        except Exception as e:
            logger.error(f"Error getting pending detections: {str(e)}")
            return Response({
                'success': False,
                'data': {},
                'message': 'Error retrieving pending detections.',
                'errors': [str(e)]
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    @action(detail=False, methods=['post'])
    def approve_detection(self, request):
        """Approve a detection and create database record"""
        try:
            detection_id = request.data.get('detection_id')
            reviewer_notes = request.data.get('notes', '')
            
            if not detection_id:
                return Response({
                    'success': False,
                    'data': {},
                    'message': 'Detection ID is required.',
                    'errors': ['Missing detection_id']
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Find the detection metadata
            detection_metadata = self._find_detection_metadata(detection_id)
            if not detection_metadata:
                return Response({
                    'success': False,
                    'data': {},
                    'message': 'Detection not found.',
                    'errors': ['Invalid detection_id']
                }, status=status.HTTP_404_NOT_FOUND)
            
            # Create alert record in database
            with transaction.atomic():
                alert = self._create_alert_from_detection(detection_metadata, request.user, reviewer_notes)
                
                if alert:
                    # Create review record
                    AlertReview.objects.create(
                        alert=alert,
                        reviewer=request.user,
                        action='confirmed',
                        notes=reviewer_notes
                    )
                    
                    # Move video files to confirmed directory
                    self._move_detection_to_confirmed(detection_metadata)
                    
                    # Send notification to end user about confirmed alert
                    self._send_confirmed_alert_notification(alert)
                    
                    logger.info(f"Detection {detection_id} approved by reviewer {request.user.id}")
                    
                    return Response({
                        'success': True,
                        'data': AlertSerializer(alert).data,
                        'message': 'Detection approved and alert created successfully.',
                        'errors': []
                    })
                else:
                    return Response({
                        'success': False,
                        'data': {},
                        'message': 'Failed to create alert from detection.',
                        'errors': ['Alert creation failed']
                    }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                    
        except Exception as e:
            logger.error(f"Error approving detection: {str(e)}")
            return Response({
                'success': False,
                'data': {},
                'message': 'Error approving detection.',
                'errors': [str(e)]
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    @action(detail=False, methods=['post'])
    def reject_detection(self, request):
        """Reject a detection and move to rejected folder"""
        try:
            detection_id = request.data.get('detection_id')
            reviewer_notes = request.data.get('notes', '')
            rejection_reason = request.data.get('reason', 'false_positive')
            
            if not detection_id:
                return Response({
                    'success': False,
                    'data': {},
                    'message': 'Detection ID is required.',
                    'errors': ['Missing detection_id']
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Find the detection metadata
            detection_metadata = self._find_detection_metadata(detection_id)
            if not detection_metadata:
                return Response({
                    'success': False,
                    'data': {},
                    'message': 'Detection not found.',
                    'errors': ['Invalid detection_id']
                }, status=status.HTTP_404_NOT_FOUND)
            
            # Move detection to rejected directory
            self._move_detection_to_rejected(detection_metadata, rejection_reason, reviewer_notes, request.user)
            
            logger.info(f"Detection {detection_id} rejected by reviewer {request.user.id}")
            
            return Response({
                'success': True,
                'data': {},
                'message': 'Detection rejected successfully.',
                'errors': []
            })
            
        except Exception as e:
            logger.error(f"Error rejecting detection: {str(e)}")
            return Response({
                'success': False,
                'data': {},
                'message': 'Error rejecting detection.',
                'errors': [str(e)]
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    @action(detail=False, methods=['post'])
    def bulk_approve(self, request):
        """Bulk approve multiple detections"""
        try:
            detection_ids = request.data.get('detection_ids', [])
            reviewer_notes = request.data.get('notes', '')
            
            if not detection_ids or not isinstance(detection_ids, list):
                return Response({
                    'success': False,
                    'data': {},
                    'message': 'Detection IDs list is required.',
                    'errors': ['Missing or invalid detection_ids']
                }, status=status.HTTP_400_BAD_REQUEST)
            
            approved_count = 0
            failed_count = 0
            created_alerts = []
            
            for detection_id in detection_ids:
                try:
                    detection_metadata = self._find_detection_metadata(detection_id)
                    if detection_metadata:
                        with transaction.atomic():
                            alert = self._create_alert_from_detection(detection_metadata, request.user, reviewer_notes)
                            if alert:
                                AlertReview.objects.create(
                                    alert=alert,
                                    reviewer=request.user,
                                    action='confirmed',
                                    notes=f"Bulk approval - {reviewer_notes}"
                                )
                                self._move_detection_to_confirmed(detection_metadata)
                                self._send_confirmed_alert_notification(alert)
                                created_alerts.append(alert)
                                approved_count += 1
                            else:
                                failed_count += 1
                    else:
                        failed_count += 1
                        
                except Exception as e:
                    logger.error(f"Error in bulk approval for detection {detection_id}: {str(e)}")
                    failed_count += 1
            
            return Response({
                'success': True,
                'data': {
                    'approved_count': approved_count,
                    'failed_count': failed_count,
                    'created_alerts': [alert.id for alert in created_alerts]
                },
                'message': f'Bulk approval completed. {approved_count} approved, {failed_count} failed.',
                'errors': []
            })
            
        except Exception as e:
            logger.error(f"Error in bulk approval: {str(e)}")
            return Response({
                'success': False,
                'data': {},
                'message': 'Error in bulk approval.',
                'errors': [str(e)]
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    @action(detail=False, methods=['get'])
    def detection_statistics(self, request):
        """Get statistics about pending detections"""
        try:
            pending_detections = detection_manager.get_pending_detections()
            
            # Calculate statistics
            stats = {
                'total_pending': len(pending_detections),
                'by_type': {},
                'by_user': {},
                'by_severity': {},
                'oldest_detection': None
            }
            
            for detection in pending_detections:
                # Count by type
                alert_type = detection.get('alert_type', 'unknown')
                stats['by_type'][alert_type] = stats['by_type'].get(alert_type, 0) + 1
                
                # Count by user
                user_email = detection.get('user_email', 'unknown')
                stats['by_user'][user_email] = stats['by_user'].get(user_email, 0) + 1
                
                # Count by severity
                severity = detection.get('severity', 'unknown')
                stats['by_severity'][severity] = stats['by_severity'].get(severity, 0) + 1
                
                # Find oldest detection
                detection_time = detection.get('detection_time', '')
                if not stats['oldest_detection'] or detection_time < stats['oldest_detection']:
                    stats['oldest_detection'] = detection_time
            
            return Response({
                'success': True,
                'data': stats,
                'message': 'Detection statistics retrieved successfully.',
                'errors': []
            })
            
        except Exception as e:
            logger.error(f"Error getting detection statistics: {str(e)}")
            return Response({
                'success': False,
                'data': {},
                'message': 'Error retrieving detection statistics.',
                'errors': [str(e)]
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def _find_detection_metadata(self, detection_id):
        """Find detection metadata by ID"""
        try:
            pending_detections = detection_manager.get_pending_detections()
            for detection in pending_detections:
                if detection.get('detection_id') == detection_id:
                    return detection
            return None
        except Exception as e:
            logger.error(f"Error finding detection metadata: {str(e)}")
            return None
    
    def _create_alert_from_detection(self, detection_metadata, reviewer, notes):
        """Create Alert database record from detection metadata"""
        try:
            from cameras.models import Camera
            
            # Get camera object
            camera = Camera.objects.get(id=detection_metadata['camera_id'])
            
            # Get relative paths for database storage
            video_path = detection_metadata.get('video_path', '')
            thumbnail_path = detection_metadata.get('thumbnail_path', '')
            
            video_relative_path = None
            thumbnail_relative_path = None
            
            if video_path and os.path.exists(video_path):
                video_relative_path = os.path.relpath(video_path, settings.MEDIA_ROOT)
            
            if thumbnail_path and os.path.exists(thumbnail_path):
                thumbnail_relative_path = os.path.relpath(thumbnail_path, settings.MEDIA_ROOT)
            
            # Create alert
            alert = Alert.objects.create(
                title=f"{detection_metadata['alert_type'].replace('_', ' ').title()} Detection - CONFIRMED",
                description=f"Confirmed detection of {detection_metadata['alert_type'].replace('_', ' ')} from camera {detection_metadata['camera_name']} with {detection_metadata['confidence']:.2f} confidence. Reviewed and approved by {reviewer.full_name}.",
                alert_type=detection_metadata['alert_type'],
                severity=detection_metadata['severity'],
                confidence=detection_metadata['confidence'],
                camera=camera,
                location=detection_metadata['camera_name'],
                video_file=video_relative_path,
                thumbnail=thumbnail_relative_path,
                status='confirmed',
                reviewed_by=reviewer,
                reviewed_at=timezone.now(),
                reviewer_notes=notes,
                resolved_by=reviewer,
                resolved_time=timezone.now(),
                detection_time=timezone.now()  # Use current time as detection time
            )
            
            logger.info(f"Created alert {alert.id} from detection {detection_metadata['detection_id']}")
            return alert
            
        except Exception as e:
            logger.error(f"Error creating alert from detection: {str(e)}")
            return None
    
    def _move_detection_to_confirmed(self, detection_metadata):
        """Move detection files to confirmed directory"""
        try:
            detection_id = detection_metadata['detection_id']
            user_id = detection_metadata['user_id']
            alert_type = detection_metadata['alert_type']
            
            # Create confirmed directory
            confirmed_dir = os.path.join(
                settings.MEDIA_ROOT, 'confirmed_detections',
                f"user_{user_id}", alert_type
            )
            os.makedirs(confirmed_dir, exist_ok=True)
            
            # Move video file
            old_video_path = detection_metadata.get('video_path')
            if old_video_path and os.path.exists(old_video_path):
                new_video_path = os.path.join(confirmed_dir, os.path.basename(old_video_path))
                os.rename(old_video_path, new_video_path)
                logger.info(f"Moved video file to: {new_video_path}")
            
            # Move thumbnail file
            old_thumbnail_path = detection_metadata.get('thumbnail_path')
            if old_thumbnail_path and os.path.exists(old_thumbnail_path):
                new_thumbnail_path = os.path.join(confirmed_dir, os.path.basename(old_thumbnail_path))
                os.rename(old_thumbnail_path, new_thumbnail_path)
                logger.info(f"Moved thumbnail file to: {new_thumbnail_path}")
            
            # Remove metadata file from pending
            metadata_files = [
                f for f in os.listdir(os.path.dirname(old_video_path or old_thumbnail_path))
                if f.startswith(detection_id) and f.endswith('_metadata.json')
            ]
            for metadata_file in metadata_files:
                metadata_path = os.path.join(os.path.dirname(old_video_path or old_thumbnail_path), metadata_file)
                if os.path.exists(metadata_path):
                    os.remove(metadata_path)
                    logger.info(f"Removed metadata file: {metadata_path}")
            
        except Exception as e:
            logger.error(f"Error moving detection to confirmed: {str(e)}")
    
    def _move_detection_to_rejected(self, detection_metadata, reason, notes, reviewer):
        """Move detection files to rejected directory"""
        try:
            detection_id = detection_metadata['detection_id']
            user_id = detection_metadata['user_id']
            alert_type = detection_metadata['alert_type']
            
            # Create rejected directory
            rejected_dir = os.path.join(
                settings.MEDIA_ROOT, 'rejected_detections',
                f"user_{user_id}", alert_type
            )
            os.makedirs(rejected_dir, exist_ok=True)
            
            # Update metadata with rejection info
            detection_metadata['rejection_reason'] = reason
            detection_metadata['rejection_notes'] = notes
            detection_metadata['rejected_by'] = reviewer.full_name
            detection_metadata['rejected_at'] = timezone.now().isoformat()
            
            # Save updated metadata in rejected directory
            rejected_metadata_path = os.path.join(rejected_dir, f"{detection_id}_rejected_metadata.json")
            with open(rejected_metadata_path, 'w') as f:
                json.dump(detection_metadata, f, indent=2)
            
            # Move video file
            old_video_path = detection_metadata.get('video_path')
            if old_video_path and os.path.exists(old_video_path):
                new_video_path = os.path.join(rejected_dir, os.path.basename(old_video_path))
                os.rename(old_video_path, new_video_path)
                logger.info(f"Moved rejected video file to: {new_video_path}")
            
            # Move thumbnail file
            old_thumbnail_path = detection_metadata.get('thumbnail_path')
            if old_thumbnail_path and os.path.exists(old_thumbnail_path):
                new_thumbnail_path = os.path.join(rejected_dir, os.path.basename(old_thumbnail_path))
                os.rename(old_thumbnail_path, new_thumbnail_path)
                logger.info(f"Moved rejected thumbnail file to: {new_thumbnail_path}")
            
            # Remove metadata file from pending
            metadata_files = [
                f for f in os.listdir(os.path.dirname(old_video_path or old_thumbnail_path))
                if f.startswith(detection_id) and f.endswith('_metadata.json')
            ]
            for metadata_file in metadata_files:
                metadata_path = os.path.join(os.path.dirname(old_video_path or old_thumbnail_path), metadata_file)
                if os.path.exists(metadata_path):
                    os.remove(metadata_path)
                    logger.info(f"Removed pending metadata file: {metadata_path}")
            
        except Exception as e:
            logger.error(f"Error moving detection to rejected: {str(e)}")
    
    def _send_confirmed_alert_notification(self, alert):
        """Send notification to end user about confirmed alert"""
        try:
            from notifications.models import NotificationSetting, NotificationLog
            from django.core.mail import send_mail
            from django.conf import settings
            
            user = alert.camera.user
            
            # Get user's notification settings
            try:
                notification_settings = NotificationSetting.objects.get(user=user)
            except NotificationSetting.DoesNotExist:
                # Create default settings
                notification_settings = NotificationSetting.objects.create(user=user)
            
            # Check if notifications should be sent
            should_send = alert._should_send_notification(notification_settings)
            
            if should_send:
                title = f"CONFIRMED ALERT: {alert.get_alert_type_display()} Detected"
                message = f"""
                A {alert.get_alert_type_display()} alert has been confirmed by our security team:
                
                Camera: {alert.camera.name}
                Time: {alert.detection_time.strftime('%Y-%m-%d %H:%M:%S')}
                Confidence: {alert.confidence:.2f}
                Severity: {alert.get_severity_display()}
                
                This alert has been verified and requires your attention.
                """
                
                # Send email notification
                if notification_settings.email_enabled:
                    try:
                        notification_log = NotificationLog.objects.create(
                            user=user,
                            title=title,
                            message=message,
                            notification_type='email',
                            alert=alert,
                            status='pending'
                        )
                        
                        send_mail(
                            subject=title,
                            message=message,
                            from_email=settings.DEFAULT_FROM_EMAIL,
                            recipient_list=[user.email],
                            fail_silently=False,
                        )
                        
                        notification_log.status = 'sent'
                        notification_log.sent_at = timezone.now()
                        notification_log.save()
                        
                        logger.info(f"Confirmation notification sent to {user.email}")
                        
                    except Exception as e:
                        if 'notification_log' in locals():
                            notification_log.status = 'failed'
                            notification_log.error_message = str(e)
                            notification_log.save()
                        logger.error(f"Error sending confirmation notification: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error sending confirmed alert notification: {str(e)}")