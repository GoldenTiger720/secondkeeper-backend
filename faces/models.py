from django.db import models
from django.contrib.auth import get_user_model
from django.urls import reverse

User = get_user_model()

class AuthorizedFace(models.Model):
    """Model for storing authorized faces for facial recognition."""
    
    ROLE_CHOICES = [
        ('primary', 'Primary'),
        ('caregiver', 'Caregiver'),
        ('family', 'Family'),
        ('other', 'Other'),
    ]
    
    # Basic information
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True, null=True)
    
    # Face image data
    face_image = models.ImageField(
        upload_to='faces/images/',
        null=True,
        blank=True,
        help_text='Upload a clear face image for recognition'
    )
    face_encoding = models.BinaryField(blank=True, null=True)  # Store face encoding as binary data
    
    # Additional information
    role = models.CharField(
        max_length=100,
        choices=ROLE_CHOICES,
        default='other'
    )
    access_level = models.CharField(max_length=50, blank=True, null=True)
    
    # Foreign keys
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='authorized_faces')
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Status
    is_active = models.BooleanField(default=True)
    
    class Meta:
        verbose_name = 'Authorized Face'
        verbose_name_plural = 'Authorized Faces'
        ordering = ['name']
        unique_together = ['user', 'name']  # Ensure unique names per user
    
    def __str__(self):
        return f"{self.name} ({self.role})"
    
    @property
    def image_url(self):
        """Get the full URL for the face image."""
        if self.face_image:
            return self.face_image.url
        return None
    
    def delete(self, *args, **kwargs):
        """Override delete to clean up image files."""
        if self.face_image:
            try:
                self.face_image.delete(save=False)
            except Exception:
                pass  # File might not exist
        super().delete(*args, **kwargs)

class FaceVerificationLog(models.Model):
    """Model for storing face verification logs."""
    
    # Verification result
    authorized_face = models.ForeignKey(
        AuthorizedFace, on_delete=models.SET_NULL, 
        null=True, blank=True, related_name='verification_logs'
    )
    is_match = models.BooleanField(default=False)
    confidence = models.FloatField(default=0.0)
    
    # Source information
    source_image = models.ImageField(upload_to='faces/verification/', blank=True, null=True)
    source_camera = models.ForeignKey(
        'cameras.Camera', on_delete=models.SET_NULL, 
        null=True, blank=True, related_name='face_verifications'
    )
    
    # Timestamp
    verified_at = models.DateTimeField(auto_now_add=True)
    
    # Metadata
    notes = models.TextField(blank=True, null=True)
    
    class Meta:
        verbose_name = 'Face Verification Log'
        verbose_name_plural = 'Face Verification Logs'
        ordering = ['-verified_at']
    
    def __str__(self):
        if self.authorized_face:
            status = "✓" if self.is_match else "✗"
            return f"{status} {self.authorized_face.name} ({self.confidence:.2f}) - {self.verified_at}"
        return f"Unknown face verification at {self.verified_at}"