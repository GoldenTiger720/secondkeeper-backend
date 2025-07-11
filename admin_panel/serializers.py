from rest_framework import serializers
from django.contrib.auth import get_user_model
from .models import SystemCheck, SystemSetting, SubscriptionPlan, UserSubscription
from accounts.serializers import UserSerializer

User = get_user_model()

class SystemCheckSerializer(serializers.ModelSerializer):
    """Serializer for system diagnostic checks."""
    
    class Meta:
        model = SystemCheck
        fields = '__all__'
        read_only_fields = ('id', 'created_at')

class SystemSettingSerializer(serializers.ModelSerializer):
    """Serializer for system settings."""
    
    class Meta:
        model = SystemSetting
        fields = '__all__'
        read_only_fields = ('id', 'created_at', 'updated_at', 'updated_by')

class SubscriptionPlanSerializer(serializers.ModelSerializer):
    """Serializer for subscription plans."""
    
    class Meta:
        model = SubscriptionPlan
        fields = '__all__'
        read_only_fields = ('id', 'created_at', 'updated_at')

class UserSubscriptionSerializer(serializers.ModelSerializer):
    """Serializer for user subscriptions."""
    
    plan_details = SubscriptionPlanSerializer(source='plan', read_only=True)
    user_details = UserSerializer(source='user', read_only=True)
    is_trial = serializers.BooleanField(read_only=True)
    is_active = serializers.BooleanField(read_only=True)
    
    class Meta:
        model = UserSubscription
        fields = '__all__'
        read_only_fields = ('id', 'created_at', 'updated_at')

class UserAdminSerializer(serializers.ModelSerializer):
    """Serializer for admin management of users."""
    
    subscription_details = UserSubscriptionSerializer(source='subscription', read_only=True)
    cameras_count = serializers.IntegerField(read_only=True)
    alerts_count = serializers.IntegerField(read_only=True)
    
    class Meta:
        model = User
        fields = (
            'id', 'email', 'full_name', 'role', 'phone_number',
            'profile_picture', 'date_joined', 'last_login', 'is_active',
            'status', 'subscription_details', 'cameras_count', 'alerts_count'
        )
        read_only_fields = ('id', 'date_joined', 'last_login', 'cameras_count', 'alerts_count')

class UserUpdateAdminSerializer(serializers.ModelSerializer):
    """Serializer for admin to update user details."""
    
    class Meta:
        model = User
        fields = (
            'email', 'full_name', 'role', 
            'phone_number', 'is_active', 'status'
        )
    
    def validate_role(self, value):
        """Validate role changes based on current user's permissions."""
        request = self.context.get('request')
        if not request:
            return value
            
        current_user = request.user
        target_user = self.instance
        
        # Only admins can change roles to admin
        if value == 'admin' and not current_user.is_admin():
            raise serializers.ValidationError("Only admins can assign admin role.")
        
        # Managers cannot change roles of other managers, reviewers, or admins
        if current_user.is_manager() and not current_user.is_admin():
            if target_user and target_user.role != 'user':
                raise serializers.ValidationError("Managers can only manage regular users.")
            if value != 'user':
                raise serializers.ValidationError("Managers can only assign user role.")
        
        return value

class CameraAdminSerializer(serializers.ModelSerializer):
    """Serializer for admin camera management."""
    
    user_name = serializers.CharField(source='user.full_name', read_only=True)
    user_email = serializers.CharField(source='user.email', read_only=True)
    
    class Meta:
        model = 'cameras.Camera'
        fields = (
            'id', 'name', 'stream_url', 'status', 'user_id', 
            'user_name', 'user_email', 'created_at', 'updated_at'
        )
        read_only_fields = ('id', 'user_name', 'user_email', 'created_at', 'updated_at')

class SystemStatusSerializer(serializers.Serializer):
    """Serializer for system status metrics."""
    
    total_users = serializers.IntegerField()
    active_users = serializers.IntegerField()
    total_cameras = serializers.IntegerField()
    online_cameras = serializers.IntegerField()
    offline_cameras = serializers.IntegerField()
    total_alerts = serializers.IntegerField()
    new_alerts = serializers.IntegerField()
    alerts_today = serializers.IntegerField()
    alerts_this_week = serializers.IntegerField()
    alert_types = serializers.DictField(child=serializers.IntegerField())
    cpu_usage = serializers.FloatField()
    memory_usage = serializers.FloatField()
    disk_usage = serializers.FloatField()
    system_health = serializers.CharField()
    uptime = serializers.CharField()
    last_backup = serializers.DateTimeField()
    system_version = serializers.CharField()