from django.db import models

class TrainingFire(models.Model):
    IMAGE_TYPE_CHOICES = [
        ('Fire', 'Fire'),
        ('NonFire', 'NonFire'),
    ]
    
    image_type = models.CharField(max_length=10, choices=IMAGE_TYPE_CHOICES)
    image_url = models.CharField(max_length=500)
    
    class Meta:
        db_table = 'training_fire'

class TrainingChoking(models.Model):
    IMAGE_TYPE_CHOICES = [
        ('Choking', 'Choking'),
        ('NonChoking', 'NonChoking'),
    ]
    
    image_type = models.CharField(max_length=15, choices=IMAGE_TYPE_CHOICES)
    image_url = models.CharField(max_length=500)
    
    class Meta:
        db_table = 'training_choking'

class TrainingFall(models.Model):
    IMAGE_TYPE_CHOICES = [
        ('Fall', 'Fall'),
        ('NonFall', 'NonFall'),
    ]
    
    image_type = models.CharField(max_length=10, choices=IMAGE_TYPE_CHOICES)
    image_url = models.CharField(max_length=500)
    
    class Meta:
        db_table = 'training_fall'

class TrainingViolence(models.Model):
    IMAGE_TYPE_CHOICES = [
        ('Violence', 'Violence'),
        ('NonViolence', 'NonViolence'),
    ]
    
    image_type = models.CharField(max_length=15, choices=IMAGE_TYPE_CHOICES)
    image_url = models.CharField(max_length=500)
    
    class Meta:
        db_table = 'training_violence'

class TrainingResult(models.Model):
    TRAINING_TYPE_CHOICES = [
        ('fire', 'Fire'),
        ('choking', 'Choking'),
        ('fall', 'Fall'),
        ('violence', 'Violence'),
    ]
    
    training_type = models.CharField(max_length=10, choices=TRAINING_TYPE_CHOICES)
    accuracy = models.FloatField()
    loss = models.FloatField()
    success = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'training_result'