# faces/migrations/0002_update_authorized_face_fields.py

from django.db import migrations, models

class Migration(migrations.Migration):

    dependencies = [
        ('faces', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='authorizedface',
            name='role',
            field=models.CharField(
                max_length=100,
                choices=[
                    ('primary', 'Primary'),
                    ('caregiver', 'Caregiver'),
                    ('family', 'Family'),
                    ('other', 'Other')
                ],
                default='other'
            ),
        ),
        migrations.AlterField(
            model_name='authorizedface',
            name='face_image',
            field=models.ImageField(
                upload_to='faces/images/',
                null=True,
                blank=True,
                help_text='Upload a clear face image for recognition'
            ),
        ),
        migrations.AddField(
            model_name='authorizedface',
            name='image_url',
            field=models.URLField(blank=True, null=True, help_text='Full URL to the face image'),
        ),
    ]