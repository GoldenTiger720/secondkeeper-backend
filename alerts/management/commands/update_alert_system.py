# alerts/management/commands/update_alert_system.py

from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils import timezone
from alerts.models import Alert
import logging

logger = logging.getLogger('security_ai')

class Command(BaseCommand):
    help = 'Update existing alerts for the new reviewer workflow system'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be updated without making changes',
        )
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force update even if alerts already have the new status',
        )
    
    def handle(self, *args, **options):
        dry_run = options['dry_run']
        force = options['force']
        
        self.stdout.write(
            self.style.SUCCESS('Starting alert system update for reviewer workflow...')
        )
        
        if dry_run:
            self.stdout.write(
                self.style.WARNING('DRY RUN MODE - No changes will be made')
            )
        
        try:
            with transaction.atomic():
                # Update existing alerts that have old status values
                alerts_to_update = Alert.objects.filter(
                    status__in=['new'] if not force else ['new', 'confirmed', 'dismissed', 'false_positive']
                )
                
                total_alerts = alerts_to_update.count()
                self.stdout.write(f"Found {total_alerts} alerts to update")
                
                if total_alerts == 0:
                    self.stdout.write(
                        self.style.SUCCESS('No alerts need updating.')
                    )
                    return
                
                updated_count = 0
                
                for alert in alerts_to_update:
                    old_status = alert.status
                    
                    # Map old statuses to new workflow
                    if alert.status == 'new':
                        new_status = 'pending_review'
                    else:
                        new_status = alert.status  # Keep existing status
                    
                    if not dry_run:
                        alert.status = new_status
                        alert.save(update_fields=['status'])
                    
                    updated_count += 1
                    
                    self.stdout.write(
                        f"Alert {alert.id}: {old_status} -> {new_status}"
                    )
                
                if not dry_run:
                    self.stdout.write(
                        self.style.SUCCESS(
                            f'Successfully updated {updated_count} alerts for reviewer workflow.'
                        )
                    )
                else:
                    self.stdout.write(
                        self.style.WARNING(
                            f'Would update {updated_count} alerts (dry run mode).'
                        )
                    )
                
                # Update camera detection settings to use new thresholds
                self._update_camera_settings(dry_run)
                
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error updating alerts: {str(e)}')
            )
            logger.error(f"Error in update_alert_system command: {str(e)}")
    
    def _update_camera_settings(self, dry_run):
        """Update camera detection settings with new optimized thresholds."""
        from cameras.models import Camera
        
        self.stdout.write('\nUpdating camera detection settings...')
        
        # New optimized settings based on detection type
        detection_settings = {
            'fire_smoke': {'conf': 0.5, 'iou': 0.45, 'size': 640},
            'fall': {'conf': 0.4, 'iou': 0.37, 'size': 512},
            'violence': {'conf': 0.35, 'iou': 0.35, 'size': 736},
            'choking': {'conf': 0.05, 'iou': 0.30, 'size': 640},
        }
        
        cameras = Camera.objects.all()
        updated_cameras = 0
        
        for camera in cameras:
            # Update to use more conservative defaults if not already set
            if camera.confidence_threshold == 0.6:  # Default from old system
                if not dry_run:
                    camera.confidence_threshold = 0.3  # More sensitive default
                    camera.save(update_fields=['confidence_threshold'])
                updated_cameras += 1
                self.stdout.write(f"Camera {camera.id}: Updated confidence threshold to 0.3")
        
        if not dry_run:
            self.stdout.write(
                self.style.SUCCESS(f'Updated {updated_cameras} camera settings.')
            )
        else:
            self.stdout.write(
                self.style.WARNING(f'Would update {updated_cameras} camera settings (dry run).')
            )
    
    def _create_sample_reviewer_assignments(self, dry_run):
        """Create sample reviewer assignments for demonstration."""
        from alerts.models import ReviewerAssignment
        from django.contrib.auth import get_user_model
        
        User = get_user_model()
        
        self.stdout.write('\nCreating sample reviewer assignments...')
        
        # Get reviewers and admins
        reviewers = User.objects.filter(
            role__in=['reviewer', 'admin'],
            is_active=True
        )
        
        if not reviewers.exists():
            self.stdout.write(
                self.style.WARNING('No reviewers found. Please create users with reviewer role first.')
            )
            return
        
        assignments_created = 0
        
        for reviewer in reviewers:
            # Check if assignment already exists
            if ReviewerAssignment.objects.filter(reviewer=reviewer).exists():
                continue
            
            if not dry_run:
                # Create assignment for all alert types
                ReviewerAssignment.objects.create(
                    reviewer=reviewer,
                    alert_types=['fire_smoke', 'fall', 'violence', 'choking', 'unauthorized_face'],
                    priority_level=2 if reviewer.role == 'admin' else 1,
                    is_active=True
                )
            
            assignments_created += 1
            self.stdout.write(f"Created assignment for reviewer: {reviewer.full_name}")
        
        if not dry_run:
            self.stdout.write(
                self.style.SUCCESS(f'Created {assignments_created} reviewer assignments.')
            )
        else:
            self.stdout.write(
                self.style.WARNING(f'Would create {assignments_created} reviewer assignments (dry run).')
            )