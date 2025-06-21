# alerts/migrations/0002_add_reviewer_workflow.py

from django.db import migrations, models
import django.db.models.deletion
from django.conf import settings


class Migration(migrations.Migration):

    dependencies = [
        ('alerts', '0001_initial'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        # Update Alert model status choices
        migrations.AlterField(
            model_name='alert',
            name='status',
            field=models.CharField(
                choices=[
                    ('pending_review', 'Pending Review'),
                    ('confirmed', 'Confirmed'),
                    ('dismissed', 'Dismissed'),
                    ('false_positive', 'False Positive')
                ],
                default='pending_review',
                max_length=20
            ),
        ),
        
        # Add reviewer workflow fields to Alert
        migrations.AddField(
            model_name='alert',
            name='reviewed_by',
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name='reviewed_alerts',
                to=settings.AUTH_USER_MODEL
            ),
        ),
        migrations.AddField(
            model_name='alert',
            name='reviewed_at',
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='alert',
            name='reviewer_notes',
            field=models.TextField(blank=True, null=True),
        ),
        
        # Create AlertReview model
        migrations.CreateModel(
            name='AlertReview',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('action', models.CharField(
                    choices=[
                        ('confirmed', 'Confirmed'),
                        ('dismissed', 'Dismissed'),
                        ('false_positive', 'Marked as False Positive'),
                        ('escalated', 'Escalated')
                    ],
                    max_length=20
                )),
                ('notes', models.TextField(blank=True, null=True)),
                ('review_time', models.DateTimeField(auto_now_add=True)),
                ('alert', models.ForeignKey(
                    on_delete=django.db.models.deletion.CASCADE,
                    related_name='review_history',
                    to='alerts.alert'
                )),
                ('reviewer', models.ForeignKey(
                    on_delete=django.db.models.deletion.CASCADE,
                    related_name='alert_reviews',
                    to=settings.AUTH_USER_MODEL
                )),
            ],
            options={
                'verbose_name': 'Alert Review',
                'verbose_name_plural': 'Alert Reviews',
                'ordering': ['-review_time'],
            },
        ),
        
        # Create ReviewerAssignment model
        migrations.CreateModel(
            name='ReviewerAssignment',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('alert_types', models.JSONField(default=list, help_text='List of alert types this reviewer handles')),
                ('is_active', models.BooleanField(default=True)),
                ('priority_level', models.IntegerField(default=1, help_text='Higher numbers = higher priority')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('reviewer', models.ForeignKey(
                    on_delete=django.db.models.deletion.CASCADE,
                    related_name='reviewer_assignments',
                    to=settings.AUTH_USER_MODEL
                )),
                ('assigned_users', models.ManyToManyField(
                    blank=True,
                    help_text='Specific users whose alerts this reviewer handles',
                    related_name='assigned_reviewers',
                    to=settings.AUTH_USER_MODEL
                )),
            ],
            options={
                'verbose_name': 'Reviewer Assignment',
                'verbose_name_plural': 'Reviewer Assignments',
                'ordering': ['-priority_level', 'created_at'],
            },
        ),
    ]