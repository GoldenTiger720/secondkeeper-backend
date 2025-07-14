from rest_framework import serializers
from alerts.models import Alert, AlertReview
from cameras.serializers import CameraSerializer
from accounts.serializers import UserSerializer
from .models import TrainingFire, TrainingChoking, TrainingFall, TrainingViolence, TrainingResult


class AlertReviewTrainingDataSerializer(serializers.ModelSerializer):
    # Reviewer information
    reviewer_name = serializers.CharField(source='reviewer.full_name', read_only=True)
    reviewer_email = serializers.CharField(source='reviewer.email', read_only=True)
    
    # Alert information
    alert_id = serializers.IntegerField(source='alert.id', read_only=True)
    alert_title = serializers.CharField(source='alert.title', read_only=True)
    alert_description = serializers.CharField(source='alert.description', read_only=True)
    alert_type = serializers.CharField(source='alert.alert_type', read_only=True)
    alert_status = serializers.CharField(source='alert.status', read_only=True)
    alert_severity = serializers.CharField(source='alert.severity', read_only=True)
    alert_confidence = serializers.FloatField(source='alert.confidence', read_only=True)
    detection_time = serializers.DateTimeField(source='alert.detection_time', read_only=True)
    resolved_time = serializers.DateTimeField(source='alert.resolved_time', read_only=True)
    
    # Camera information
    camera_id = serializers.IntegerField(source='alert.camera.id', read_only=True)
    camera_name = serializers.CharField(source='alert.camera.name', read_only=True)
    camera_location = serializers.CharField(source='alert.location', read_only=True)
    
    # User (camera owner) information
    user_email = serializers.CharField(source='alert.camera.user.email', read_only=True)
    user_name = serializers.CharField(source='alert.camera.user.full_name', read_only=True)
    
    # Media files
    video_file = serializers.FileField(source='alert.video_file', read_only=True)
    thumbnail = serializers.ImageField(source='alert.thumbnail', read_only=True)
    
    # Additional alert metadata
    alert_notes = serializers.CharField(source='alert.notes', read_only=True)
    is_test = serializers.BooleanField(source='alert.is_test', read_only=True)
    alert_created_at = serializers.DateTimeField(source='alert.created_at', read_only=True)
    alert_updated_at = serializers.DateTimeField(source='alert.updated_at', read_only=True)
    
    class Meta:
        model = AlertReview
        fields = [
            # AlertReview fields
            'id',
            'action',
            'notes',
            'review_time',
            'reviewer',
            'reviewer_name',
            'reviewer_email',
            
            # Alert information
            'alert_id',
            'alert_title',
            'alert_description',
            'alert_type',
            'alert_status',
            'alert_severity',
            'alert_confidence',
            'detection_time',
            'resolved_time',
            
            # Camera information
            'camera_id',
            'camera_name',
            'camera_location',
            
            # User information
            'user_email',
            'user_name',
            
            # Media files
            'video_file',
            'thumbnail',
            
            # Additional metadata
            'alert_notes',
            'is_test',
            'alert_created_at',
            'alert_updated_at'
        ]


# Keep the old serializer for backward compatibility if needed
class AlertTrainingDataSerializer(serializers.ModelSerializer):
    camera = CameraSerializer(read_only=True)
    review_history = serializers.SerializerMethodField()
    resolved_by_name = serializers.CharField(source='resolved_by.full_name', read_only=True)
    resolved_by_email = serializers.CharField(source='resolved_by.email', read_only=True)
    reviewed_by_name = serializers.CharField(source='reviewed_by.full_name', read_only=True)
    reviewed_by_email = serializers.CharField(source='reviewed_by.email', read_only=True)
    user_email = serializers.CharField(source='camera.user.email', read_only=True)
    user_name = serializers.CharField(source='camera.user.full_name', read_only=True)
    
    def get_review_history(self, obj):
        reviews = obj.review_history.all()
        return AlertReviewTrainingDataSerializer(reviews, many=True).data
    
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


class TrainingFireSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrainingFire
        fields = ['id', 'image_type', 'image_url']


class TrainingChokingSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrainingChoking
        fields = ['id', 'image_type', 'image_url']


class TrainingFallSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrainingFall
        fields = ['id', 'image_type', 'image_url']


class TrainingViolenceSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrainingViolence
        fields = ['id', 'image_type', 'image_url']


class TrainingResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrainingResult
        fields = ['id', 'training_type', 'accuracy', 'loss', 'success', 'created_at']