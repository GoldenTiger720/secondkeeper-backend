# alerts/serializers.py - Updated with reviewer workflow serializers

from rest_framework import serializers
from django.contrib.auth import get_user_model
from .models import Alert, AlertReview, ReviewerAssignment
from accounts.serializers import UserSerializer
from cameras.serializers import CameraListSerializer

User = get_user_model()

class AlertSerializer(serializers.ModelSerializer):
    """Serializer for Alert model with reviewer workflow fields."""
    
    camera_details = CameraListSerializer(source='camera', read_only=True)
    resolved_by_details = UserSerializer(source='resolved_by', read_only=True)
    reviewed_by_details = UserSerializer(source='reviewed_by', read_only=True)
    time_since_detection = serializers.SerializerMethodField()
    is_pending_review = serializers.BooleanField(read_only=True)
    
    class Meta:
        model = Alert
        fields = '__all__'
        read_only_fields = ('created_at', 'updated_at', 'detection_time')
    
    def get_time_since_detection(self, obj):
        """Get time elapsed since detection in human readable format."""
        time_diff = obj.time_since_detection
        
        if time_diff.days > 0:
            return f"{time_diff.days} days ago"
        elif time_diff.seconds > 3600:
            hours = time_diff.seconds // 3600
            return f"{hours} hours ago"
        elif time_diff.seconds > 60:
            minutes = time_diff.seconds // 60
            return f"{minutes} minutes ago"
        else:
            return "Just now"

class AlertListSerializer(serializers.ModelSerializer):
    """Serializer for listing alerts with minimal information."""
    
    camera_name = serializers.CharField(source='camera.name', read_only=True)
    user_name = serializers.CharField(source='camera.user.full_name', read_only=True)
    time_since_detection = serializers.SerializerMethodField()
    
    class Meta:
        model = Alert
        fields = (
            'id', 'title', 'alert_type', 'status', 'severity',
            'confidence', 'detection_time', 'camera_name', 'user_name',
            'thumbnail', 'time_since_detection', 'is_pending_review'
        )
        read_only_fields = fields
    
    def get_time_since_detection(self, obj):
        """Get time elapsed since detection in human readable format."""
        time_diff = obj.time_since_detection
        
        if time_diff.days > 0:
            return f"{time_diff.days}d"
        elif time_diff.seconds > 3600:
            hours = time_diff.seconds // 3600
            return f"{hours}h"
        elif time_diff.seconds > 60:
            minutes = time_diff.seconds // 60
            return f"{minutes}m"
        else:
            return "now"

class AlertCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating a new alert."""
    
    class Meta:
        model = Alert
        fields = (
            'title', 'description', 'alert_type', 'severity',
            'confidence', 'camera', 'location', 'video_file',
            'thumbnail', 'notes', 'is_test'
        )

class AlertStatusUpdateSerializer(serializers.ModelSerializer):
    """Serializer for updating alert status by reviewers."""
    
    notes = serializers.CharField(required=False, allow_blank=True)
    reviewer_notes = serializers.CharField(required=False, allow_blank=True)
    
    class Meta:
        model = Alert
        fields = ('status', 'notes', 'reviewer_notes')
    
    def validate_status(self, value):
        """Validate the status field."""
        valid_statuses = ['confirmed', 'dismissed', 'false_positive']
        if value not in valid_statuses:
            raise serializers.ValidationError(
                f"Invalid status. Valid choices are: {', '.join(valid_statuses)}"
            )
        return value

class AlertReviewSerializer(serializers.ModelSerializer):
    """Serializer for alert review records."""
    
    reviewer_name = serializers.CharField(source='reviewer.full_name', read_only=True)
    alert_title = serializers.CharField(source='alert.title', read_only=True)
    alert_type = serializers.CharField(source='alert.alert_type', read_only=True)
    camera_name = serializers.CharField(source='alert.camera.name', read_only=True)
    
    class Meta:
        model = AlertReview
        fields = '__all__'
        read_only_fields = ('id', 'review_time')

class AlertReviewCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating alert reviews."""
    
    class Meta:
        model = AlertReview
        fields = ('alert', 'action', 'notes')
    
    def validate_action(self, value):
        """Validate the review action."""
        valid_actions = ['confirmed', 'dismissed', 'false_positive', 'escalated']
        if value not in valid_actions:
            raise serializers.ValidationError(
                f"Invalid action. Valid choices are: {', '.join(valid_actions)}"
            )
        return value

class ReviewerAssignmentSerializer(serializers.ModelSerializer):
    """Serializer for reviewer assignments."""
    
    reviewer_name = serializers.CharField(source='reviewer.full_name', read_only=True)
    reviewer_email = serializers.CharField(source='reviewer.email', read_only=True)
    assigned_user_names = serializers.SerializerMethodField()
    
    class Meta:
        model = ReviewerAssignment
        fields = '__all__'
        read_only_fields = ('id', 'created_at', 'updated_at')
    
    def get_assigned_user_names(self, obj):
        """Get names of assigned users."""
        return [user.full_name for user in obj.assigned_users.all()]
    
    def validate_reviewer(self, value):
        """Validate that the reviewer has appropriate role."""
        if value.role not in ['reviewer', 'admin']:
            raise serializers.ValidationError(
                "Only users with 'reviewer' or 'admin' role can be assigned as reviewers."
            )
        return value
    
    def validate_alert_types(self, value):
        """Validate alert types."""
        valid_types = ['fire_smoke', 'fall', 'violence', 'choking', 'unauthorized_face']
        for alert_type in value:
            if alert_type not in valid_types:
                raise serializers.ValidationError(
                    f"Invalid alert type: {alert_type}. Valid types are: {', '.join(valid_types)}"
                )
        return value

class ReviewerAssignmentCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating reviewer assignments."""
    
    class Meta:
        model = ReviewerAssignment
        fields = ('reviewer', 'alert_types', 'assigned_users', 'priority_level', 'is_active')
    
    def validate_reviewer(self, value):
        """Validate that the reviewer has appropriate role."""
        if value.role not in ['reviewer', 'admin']:
            raise serializers.ValidationError(
                "Only users with 'reviewer' or 'admin' role can be assigned as reviewers."
            )
        return value

class AlertSummarySerializer(serializers.Serializer):
    """Serializer for alert summary statistics."""
    
    total_alerts = serializers.IntegerField()
    pending_alerts = serializers.IntegerField()
    confirmed_alerts = serializers.IntegerField()
    dismissed_alerts = serializers.IntegerField()
    false_positive_alerts = serializers.IntegerField()
    
    by_type = serializers.DictField()
    by_severity = serializers.DictField()
    by_status = serializers.DictField()
    
    daily_count = serializers.ListField(
        child=serializers.DictField(
            child=serializers.IntegerField()
        )
    )
    weekly_count = serializers.ListField(
        child=serializers.DictField(
            child=serializers.IntegerField()
        )
    )
    monthly_count = serializers.ListField(
        child=serializers.DictField(
            child=serializers.IntegerField()
        )
    )

class ReviewerDashboardSerializer(serializers.Serializer):
    """Serializer for reviewer dashboard data."""
    
    pending_alerts = serializers.IntegerField()
    urgent_alerts = serializers.IntegerField()
    alerts_reviewed_today = serializers.IntegerField()
    avg_review_time_hours = serializers.FloatField()
    oldest_pending_hours = serializers.FloatField(allow_null=True)
    
    alerts_by_type = serializers.DictField()
    alerts_by_severity = serializers.DictField()
    recent_reviews = AlertReviewSerializer(many=True)
    
    reviewer_info = serializers.DictField()

class NotificationRequestSerializer(serializers.Serializer):
    """Serializer for sending notifications after alert confirmation."""
    
    notification_types = serializers.ListField(
        child=serializers.ChoiceField(choices=['email', 'sms', 'whatsapp']),
        default=['email']
    )
    custom_message = serializers.CharField(required=False, allow_blank=True)
    urgent = serializers.BooleanField(default=False)
    
    def validate_notification_types(self, value):
        """Validate notification types."""
        valid_types = ['email', 'sms', 'whatsapp']
        for notification_type in value:
            if notification_type not in valid_types:
                raise serializers.ValidationError(
                    f"Invalid notification type: {notification_type}"
                )
        return value