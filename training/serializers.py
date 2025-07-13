from rest_framework import serializers
from alerts.models import Alert, AlertReview
from cameras.serializers import CameraSerializer
from accounts.serializers import UserSerializer


class AlertReviewSerializer(serializers.ModelSerializer):
    reviewer_name = serializers.CharField(source='reviewer.full_name', read_only=True)
    
    class Meta:
        model = AlertReview
        fields = [
            'id',
            'action',
            'notes',
            'review_time',
            'reviewer',
            'reviewer_name'
        ]


class AlertTrainingDataSerializer(serializers.ModelSerializer):
    camera = CameraSerializer(read_only=True)
    review_history = AlertReviewSerializer(many=True, read_only=True)
    resolved_by_name = serializers.CharField(source='resolved_by.full_name', read_only=True)
    reviewed_by_name = serializers.CharField(source='reviewed_by.full_name', read_only=True)
    
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
            'reviewed_by',
            'reviewed_by_name',
            'reviewed_at',
            'reviewer_notes',
            'camera',
            'location',
            'video_file',
            'thumbnail',
            'notes',
            'is_test',
            'created_at',
            'updated_at',
            'review_history'
        ]