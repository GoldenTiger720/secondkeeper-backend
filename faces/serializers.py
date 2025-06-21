from rest_framework import serializers
from .models import AuthorizedFace, FaceVerificationLog

class AuthorizedFaceSerializer(serializers.ModelSerializer):
    """Serializer for AuthorizedFace model."""
    
    image_url = serializers.SerializerMethodField()
    
    class Meta:
        model = AuthorizedFace
        exclude = ('face_encoding',)
        read_only_fields = ('id', 'user', 'created_at', 'updated_at')
    
    def get_image_url(self, obj):
        """Get the full URL for the face image."""
        if obj.face_image:
            request = self.context.get('request')
            if request:
                return request.build_absolute_uri(obj.face_image.url)
            return obj.face_image.url
        return None

class AuthorizedFaceCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating a new authorized face."""
    
    face_image = serializers.ImageField(required=False, allow_null=True)
    
    class Meta:
        model = AuthorizedFace
        fields = ('name', 'description', 'face_image', 'role', 'access_level', 'is_active')
    
    def validate_name(self, value):
        """Validate that the name is not empty and unique for the user."""
        if not value or not value.strip():
            raise serializers.ValidationError("Name cannot be empty.")
        
        # Check for uniqueness within the user's faces
        user = self.context['request'].user
        if AuthorizedFace.objects.filter(user=user, name__iexact=value.strip()).exists():
            raise serializers.ValidationError("A person with this name already exists.")
        
        return value.strip()
    
    def validate_face_image(self, value):
        """Validate the uploaded face image."""
        if value:
            # Check file size (max 5MB)
            if value.size > 5 * 1024 * 1024:
                raise serializers.ValidationError("Image file too large. Maximum size is 5MB.")
            
            # Check file type
            if not value.content_type.startswith('image/'):
                raise serializers.ValidationError("File must be an image.")
            
            # Check supported formats
            allowed_formats = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif']
            if value.content_type not in allowed_formats:
                raise serializers.ValidationError("Unsupported image format. Use JPEG, PNG, or GIF.")
        
        return value
    
    def create(self, validated_data):
        """Create and return a new authorized face."""
        user = self.context['request'].user
        authorized_face = AuthorizedFace.objects.create(user=user, **validated_data)
        return authorized_face

class AuthorizedFaceUpdateSerializer(serializers.ModelSerializer):
    """Serializer for updating an authorized face."""
    
    face_image = serializers.ImageField(required=False, allow_null=True)
    
    class Meta:
        model = AuthorizedFace
        fields = ('name', 'description', 'face_image', 'role', 'access_level', 'is_active')
    
    def validate_name(self, value):
        """Validate that the name is not empty and unique for the user."""
        if not value or not value.strip():
            raise serializers.ValidationError("Name cannot be empty.")
        
        # Check for uniqueness within the user's faces (excluding current instance)
        user = self.context['request'].user
        queryset = AuthorizedFace.objects.filter(user=user, name__iexact=value.strip())
        
        if self.instance:
            queryset = queryset.exclude(pk=self.instance.pk)
        
        if queryset.exists():
            raise serializers.ValidationError("A person with this name already exists.")
        
        return value.strip()
    
    def validate_face_image(self, value):
        """Validate the uploaded face image."""
        if value:
            # Check file size (max 5MB)
            if value.size > 5 * 1024 * 1024:
                raise serializers.ValidationError("Image file too large. Maximum size is 5MB.")
            
            # Check file type
            if not value.content_type.startswith('image/'):
                raise serializers.ValidationError("File must be an image.")
            
            # Check supported formats
            allowed_formats = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif']
            if value.content_type not in allowed_formats:
                raise serializers.ValidationError("Unsupported image format. Use JPEG, PNG, or GIF.")
        
        return value

class FaceVerificationSerializer(serializers.ModelSerializer):
    """Serializer for face verification logs."""
    
    authorized_face_name = serializers.CharField(source='authorized_face.name', read_only=True)
    camera_name = serializers.CharField(source='source_camera.name', read_only=True)
    
    class Meta:
        model = FaceVerificationLog
        fields = '__all__'
        read_only_fields = ('id', 'verified_at')

class FaceVerificationRequestSerializer(serializers.Serializer):
    """Serializer for face verification requests."""
    
    face_image = serializers.ImageField(required=True)
    camera_id = serializers.IntegerField(required=False)
    confidence_threshold = serializers.FloatField(required=False, default=0.6, min_value=0.0, max_value=1.0)
    
    def validate_face_image(self, value):
        """Validate the uploaded face image."""
        # Check file size (max 10MB for verification)
        if value.size > 10 * 1024 * 1024:
            raise serializers.ValidationError("Image file too large. Maximum size is 10MB.")
        
        # Check file type
        if not value.content_type.startswith('image/'):
            raise serializers.ValidationError("File must be an image.")
        
        return value

class FaceVerificationResponseSerializer(serializers.Serializer):
    """Serializer for face verification responses."""
    
    is_match = serializers.BooleanField()
    confidence = serializers.FloatField()
    matched_face = AuthorizedFaceSerializer(required=False, allow_null=True)
    verification_id = serializers.IntegerField(required=False, allow_null=True)

class FaceUploadSerializer(serializers.Serializer):
    """Serializer for face upload endpoint."""
    
    name = serializers.CharField(max_length=100)
    role = serializers.ChoiceField(choices=[
        ('primary', 'Primary'),
        ('caregiver', 'Caregiver'),
        ('family', 'Family'),
        ('other', 'Other')
    ])
    face_image = serializers.ImageField()
    description = serializers.CharField(max_length=500, required=False, allow_blank=True)
    
    def validate_name(self, value):
        """Validate the name field."""
        if not value or not value.strip():
            raise serializers.ValidationError("Name cannot be empty.")
        return value.strip()
    
    def validate_face_image(self, value):
        """Validate the uploaded face image."""
        # Check file size (max 5MB)
        if value.size > 5 * 1024 * 1024:
            raise serializers.ValidationError("Image file too large. Maximum size is 5MB.")
        
        # Check file type
        if not value.content_type.startswith('image/'):
            raise serializers.ValidationError("File must be an image.")
        
        # Check supported formats
        allowed_formats = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif']
        if value.content_type not in allowed_formats:
            raise serializers.ValidationError("Unsupported image format. Use JPEG, PNG, or GIF.")
        
        return value