# alerts/models.py - Updated with reviewer workflow

from django.db import models
from django.contrib.auth import get_user_model
from django.utils import timezone
from cameras.models import Camera

User = get_user_model()

class Alert(models.Model):
    """Alert model for storing detection alerts with reviewer workflow."""
    
    ALERT_TYPE_CHOICES = (
        ('fire_smoke', 'Fire and Smoke'),
        ('fall', 'Fall Detection'),
        ('violence', 'Violence'),
        ('choking', 'Choking'),
        ('unauthorized_face', 'Unauthorized Face'),
        ('other', 'Other'),
    )
    
    STATUS_CHOICES = (
        ('pending_review', 'Pending Review'),  # New status for reviewer workflow
        ('confirmed', 'Confirmed'),
        ('dismissed', 'Dismissed'),
        ('false_positive', 'False Positive'),
    )
    
    SEVERITY_CHOICES = (
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High'),
        ('critical', 'Critical'),
    )
    
    # Basic information
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True, null=True)
    alert_type = models.CharField(max_length=20, choices=ALERT_TYPE_CHOICES)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending_review')
    severity = models.CharField(max_length=20, choices=SEVERITY_CHOICES, default='medium')
    confidence = models.FloatField(default=0.0)
    
    # Detection details
    detection_time = models.DateTimeField(auto_now_add=True)
    resolved_time = models.DateTimeField(blank=True, null=True)
    resolved_by = models.ForeignKey(
        User, on_delete=models.SET_NULL, null=True, blank=True, 
        related_name='resolved_alerts'
    )
    
    # Reviewer workflow fields
    reviewed_by = models.ForeignKey(
        User, on_delete=models.SET_NULL, null=True, blank=True,
        related_name='reviewed_alerts'
    )
    reviewed_at = models.DateTimeField(blank=True, null=True)
    reviewer_notes = models.TextField(blank=True, null=True)
    
    # Foreign keys
    camera = models.ForeignKey(Camera, on_delete=models.CASCADE, related_name='alerts')
    
    # Location data
    location = models.CharField(max_length=200, blank=True, null=True)
    
    # Media storage
    video_file = models.FileField(upload_to='alerts/videos/', blank=True, null=True)
    thumbnail = models.ImageField(upload_to='alerts/thumbnails/', blank=True, null=True)
    
    # Metadata
    notes = models.TextField(blank=True, null=True)
    is_test = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-detection_time']
        verbose_name = 'Alert'
        verbose_name_plural = 'Alerts'
    
    def __str__(self):
        return f"{self.get_alert_type_display()} Alert - {self.detection_time}"
    
    def mark_as_confirmed(self, user, send_notification=True):
        """Mark the alert as confirmed by reviewer."""
        self.status = 'confirmed'
        self.resolved_time = timezone.now()
        self.resolved_by = user
        self.reviewed_by = user
        self.reviewed_at = timezone.now()
        self.save(update_fields=['status', 'resolved_time', 'resolved_by', 'reviewed_by', 'reviewed_at', 'updated_at'])
        
        # Send notification to end user if confirmed
        if send_notification:
            self._send_user_notification()
    
    def mark_as_dismissed(self, user, notes=None):
        """Mark the alert as dismissed by reviewer."""
        self.status = 'dismissed'
        self.resolved_time = timezone.now()
        self.resolved_by = user
        self.reviewed_by = user
        self.reviewed_at = timezone.now()
        if notes:
            self.reviewer_notes = notes
        self.save(update_fields=['status', 'resolved_time', 'resolved_by', 'reviewed_by', 'reviewed_at', 'reviewer_notes', 'updated_at'])
    
    def mark_as_false_positive(self, user, notes=None):
        """Mark the alert as false positive by reviewer."""
        self.status = 'false_positive'
        self.resolved_time = timezone.now()
        self.resolved_by = user
        self.reviewed_by = user
        self.reviewed_at = timezone.now()
        if notes:
            self.reviewer_notes = notes
        self.save(update_fields=['status', 'resolved_time', 'resolved_by', 'reviewed_by', 'reviewed_at', 'reviewer_notes', 'updated_at'])
    
    def _send_user_notification(self):
        """Send notification to end user after confirmation."""
        try:
            from notifications.models import NotificationSetting, NotificationLog
            from django.core.mail import send_mail
            from django.conf import settings
            
            user = self.camera.user
            
            # Get user's notification settings
            try:
                notification_settings = NotificationSetting.objects.get(user=user)
            except NotificationSetting.DoesNotExist:
                # Create default settings
                notification_settings = NotificationSetting.objects.create(user=user)
            
            # Check if notifications should be sent
            should_send = self._should_send_notification(notification_settings)
            
            if should_send:
                title = f"CONFIRMED ALERT: {self.get_alert_type_display()} Detected"
                message = f"""
                A {self.get_alert_type_display()} alert has been confirmed by our security team:
                
                Camera: {self.camera.name}
                Time: {self.detection_time.strftime('%Y-%m-%d %H:%M:%S')}
                Confidence: {self.confidence:.2f}
                Severity: {self.get_severity_display()}
                
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
                            alert=self,
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
                        
                    except Exception as e:
                        if 'notification_log' in locals():
                            notification_log.status = 'failed'
                            notification_log.error_message = str(e)
                            notification_log.save()
                
                # TODO: Add SMS/WhatsApp notification here
                # This would require integration with services like Twilio or WhatsApp Business API
                
        except Exception as e:
            import logging
            logger = logging.getLogger('security_ai')
            logger.error(f"Error sending user notification: {str(e)}")
    
    def _should_send_notification(self, settings):
        """Check if notification should be sent based on user settings."""
        try:
            # Check if notification type is enabled for this alert type
            alert_type_enabled = getattr(
                settings, 
                f"email_for_{self.alert_type}", 
                True
            )
            if not alert_type_enabled:
                return False
            
            # Check severity threshold
            severity_rank = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
            alert_severity_rank = severity_rank.get(self.severity, 2)
            min_severity_rank = severity_rank.get(settings.min_severity_email, 2)
            
            if alert_severity_rank < min_severity_rank:
                return False
            
            # Check quiet hours
            if settings.quiet_hours_enabled:
                current_time = timezone.now().time()
                start_time = settings.quiet_hours_start
                end_time = settings.quiet_hours_end
                
                # Check if in quiet hours
                in_quiet_hours = False
                if start_time <= end_time:
                    in_quiet_hours = start_time <= current_time <= end_time
                else:
                    in_quiet_hours = current_time >= start_time or current_time <= end_time
                
                if in_quiet_hours and self.severity != 'critical':
                    return False
            
            return True
            
        except Exception as e:
            import logging
            logger = logging.getLogger('security_ai')
            logger.error(f"Error checking notification settings: {str(e)}")
            return False
    
    def add_notes(self, notes):
        """Add notes to the alert."""
        self.notes = notes
        self.save(update_fields=['notes', 'updated_at'])
    
    @property
    def is_resolved(self):
        """Check if the alert is resolved."""
        return self.status in ['confirmed', 'dismissed', 'false_positive']
    
    @property
    def is_pending_review(self):
        """Check if the alert is pending review."""
        return self.status == 'pending_review'
    
    @property
    def time_since_detection(self):
        """Get the time elapsed since detection."""
        return timezone.now() - self.detection_time


class AlertReview(models.Model):
    """Model for tracking alert review history."""
    
    REVIEW_ACTION_CHOICES = (
        ('confirmed', 'Confirmed'),
        ('dismissed', 'Dismissed'),
        ('false_positive', 'Marked as False Positive'),
        ('escalated', 'Escalated'),
    )
    
    alert = models.ForeignKey(Alert, on_delete=models.CASCADE, related_name='review_history')
    reviewer = models.ForeignKey(User, on_delete=models.CASCADE, related_name='alert_reviews')
    action = models.CharField(max_length=20, choices=REVIEW_ACTION_CHOICES)
    notes = models.TextField(blank=True, null=True)
    review_time = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-review_time']
        verbose_name = 'Alert Review'
        verbose_name_plural = 'Alert Reviews'
    
    def __str__(self):
        return f"Review of Alert {self.alert.id} by {self.reviewer.full_name} - {self.action}"


class ReviewerAssignment(models.Model):
    """Model for assigning reviewers to specific alert types or users."""
    
    reviewer = models.ForeignKey(User, on_delete=models.CASCADE, related_name='reviewer_assignments')
    alert_types = models.JSONField(default=list, help_text="List of alert types this reviewer handles")
    assigned_users = models.ManyToManyField(
        User, blank=True, related_name='assigned_reviewers',
        help_text="Specific users whose alerts this reviewer handles"
    )
    is_active = models.BooleanField(default=True)
    priority_level = models.IntegerField(default=1, help_text="Higher numbers = higher priority")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-priority_level', 'created_at']
        verbose_name = 'Reviewer Assignment'
        verbose_name_plural = 'Reviewer Assignments'
    
    def __str__(self):
        return f"{self.reviewer.full_name} - {', '.join(self.alert_types) if self.alert_types else 'All types'}"