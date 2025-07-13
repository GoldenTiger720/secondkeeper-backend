from rest_framework import serializers
from alerts.models import Alert, AlertReview
from cameras.serializers import CameraSerializer
from accounts.serializers import UserSerializer


class AlertReviewSerializer(serializers.ModelSerializer):
    reviewer_name = serializers.CharField(source='reviewer.full_name', read_only=True)
    reviewer_email = serializers.CharField(source='reviewer.email', read_only=True)
    
    class Meta:
        model = AlertReview
        fields = [
            'id',
            'action',
            'notes',
            'review_time',
            'reviewer',
            'reviewer_name',
            'reviewer_email'
        ]


class AlertTrainingDataSerializer(serializers.ModelSerializer):
    camera = CameraSerializer(read_only=True)
    review_history = AlertReviewSerializer(many=True, read_only=True)
    resolved_by_name = serializers.CharField(source='resolved_by.full_name', read_only=True)
    resolved_by_email = serializers.CharField(source='resolved_by.email', read_only=True)
    reviewed_by_name = serializers.CharField(source='reviewed_by.full_name', read_only=True)
    reviewed_by_email = serializers.CharField(source='reviewed_by.email', read_only=True)
    user_email = serializers.CharField(source='camera.user.email', read_only=True)
    user_name = serializers.CharField(source='camera.user.full_name', read_only=True)
    
    class Meta:
        model = Alert
        fields = [
            'id',
            'title',
            'description',
            'alert_type',
            'status',
            'severity',
            'confidence',
            'detection_time',
            'resolved_time',
            'resolved_by',
            'resolved_by_name',
            'resolved_by_email',
            'reviewed_by',
            'reviewed_by_name',
            'reviewed_by_email',
            'reviewed_at',
            'reviewer_notes',
            'camera',
            'user_email',
            'user_name',
            'location',
            'video_file',
            'thumbnail',
            'notes',
            'is_test',
            'created_at',
            'updated_at',
            'review_history'
        ]