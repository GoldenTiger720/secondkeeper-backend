# alerts/management/commands/migrate_to_approval_system.py

from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils import timezone
from django.conf import settings
import os
import json
import logging
from alerts.models import Alert

logger = logging.getLogger('security_ai')

class Command(BaseCommand):
    help = 'Migrate existing alerts to the new approval system and clean up old pending alerts'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be changed without making changes',
        )
        parser.add_argument(
            '--delete-pending',
            action='store_true',
            help='Delete all alerts with pending_review status',
        )
        parser.add_argument(
            '--backup-pending',
            action='store_true',
            help='Backup pending alerts to filesystem before deletion',
        )
    
    def handle(self, *args, **options):
        dry_run = options['dry_run']
        delete_pending = options['delete_pending']
        backup_pending = options['backup_pending']
        
        self.stdout.write(
            self.style.SUCCESS('Starting migration to new approval system...')
        )
        
        if dry_run:
            self.stdout.write(
                self.style.WARNING('DRY RUN MODE - No changes will be made')
            )
        
        try:
            # Step 1: Handle pending alerts
            pending_alerts = Alert.objects.filter(status='pending_review')
            pending_count = pending_alerts.count()
            
            self.stdout.write(f"Found {pending_count} alerts with pending_review status")
            
            if pending_count > 0:
                if backup_pending:
                    self._backup_pending_alerts(pending_alerts, dry_run)
                
                if delete_pending:
                    if not dry_run:
                        with transaction.atomic():
                            deleted_count = pending_alerts.delete()[0]
                            self.stdout.write(
                                self.style.SUCCESS(f'Deleted {deleted_count} pending alerts')
                            )
                    else:
                        self.stdout.write(
                            self.style.WARNING(f'Would delete {pending_count} pending alerts (dry run)')
                        )
                else:
                    self.stdout.write(
                        self.style.WARNING(
                            'Pending alerts found but not deleted. Use --delete-pending to remove them.'
                        )
                    )
            
            # Step 2: Update confirmed alerts
            confirmed_alerts = Alert.objects.filter(status='confirmed')
            confirmed_count = confirmed_alerts.count()
            
            self.stdout.write(f"Found {confirmed_count} confirmed alerts")
            
            if confirmed_count > 0 and not dry_run:
                # These alerts are already valid and don't need changes
                self.stdout.write(
                    self.style.SUCCESS(f'Confirmed alerts are already compatible with new system')
                )
            
            # Step 3: Create directory structure for new system
            self._create_directory_structure(dry_run)
            
            # Step 4: Provide instructions
            self._show_instructions()
            
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error during migration: {str(e)}')
            )
            logger.error(f"Error in migrate_to_approval_system command: {str(e)}")
    
    def _backup_pending_alerts(self, pending_alerts, dry_run):
        """Backup pending alerts to JSON files"""
        try:
            backup_dir = os.path.join(settings.MEDIA_ROOT, 'alert_backups', 'pending_alerts')
            
            if not dry_run:
                os.makedirs(backup_dir, exist_ok=True)
            
            backup_count = 0
            
            for alert in pending_alerts:
                try:
                    # Create backup data
                    backup_data = {
                        'id': alert.id,
                        'title': alert.title,
                        'description': alert.description,
                        'alert_type': alert.alert_type,
                        'status': alert.status,
                        'severity': alert.severity,
                        'confidence': alert.confidence,
                        'detection_time': alert.detection_time.isoformat(),
                        'camera_id': alert.camera.id,
                        'camera_name': alert.camera.name,
                        'user_id': alert.camera.user.id,
                        'user_email': alert.camera.user.email,
                        'location': alert.location,
                        'video_file': str(alert.video_file) if alert.video_file else None,
                        'thumbnail': str(alert.thumbnail) if alert.thumbnail else None,
                        'notes': alert.notes,
                        'reviewer_notes': alert.reviewer_notes,
                        'created_at': alert.created_at.isoformat(),
                        'updated_at': alert.updated_at.isoformat()
                    }
                    
                    # Save backup file
                    if not dry_run:
                        backup_filename = f"alert_{alert.id}_{alert.alert_type}_backup.json"
                        backup_path = os.path.join(backup_dir, backup_filename)
                        
                        with open(backup_path, 'w') as f:
                            json.dump(backup_data, f, indent=2)
                    
                    backup_count += 1
                    
                except Exception as e:
                    self.stdout.write(
                        self.style.ERROR(f'Error backing up alert {alert.id}: {str(e)}')
                    )
            
            if not dry_run:
                self.stdout.write(
                    self.style.SUCCESS(f'Backed up {backup_count} pending alerts to {backup_dir}')
                )
            else:
                self.stdout.write(
                    self.style.WARNING(f'Would backup {backup_count} pending alerts (dry run)')
                )
                
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error during backup: {str(e)}')
            )
    
    def _create_directory_structure(self, dry_run):
        """Create directory structure for new approval system"""
        try:
            directories = [
                os.path.join(settings.MEDIA_ROOT, 'pending_detections'),
                os.path.join(settings.MEDIA_ROOT, 'confirmed_detections'),
                os.path.join(settings.MEDIA_ROOT, 'rejected_detections'),
            ]
            
            created_count = 0
            
            for directory in directories:
                if not os.path.exists(directory):
                    if not dry_run:
                        os.makedirs(directory, exist_ok=True)
                    created_count += 1
            
            if not dry_run:
                self.stdout.write(
                    self.style.SUCCESS(f'Created {created_count} directories for new approval system')
                )
            else:
                self.stdout.write(
                    self.style.WARNING(f'Would create {created_count} directories (dry run)')
                )
                
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error creating directories: {str(e)}')
            )
    
    def _show_instructions(self):
        """Show post-migration instructions"""
        self.stdout.write(
            self.style.SUCCESS('\n' + '='*60)
        )
        self.stdout.write(
            self.style.SUCCESS('MIGRATION TO NEW APPROVAL SYSTEM COMPLETED')
        )
        self.stdout.write(
            self.style.SUCCESS('='*60)
        )
        
        instructions = """
Next steps:

1. UPDATE YOUR DETECTION MANAGER:
   - The detection system now saves videos to filesystem only
   - No database records are created until reviewer approval
   - Videos are saved with bounding boxes and detection info

2. RESTART DETECTION SERVICE:
   - Run: python manage.py start_detection
   - The service will now use the new video-only mode

3. REVIEWER WORKFLOW:
   - Reviewers access: /api/alerts/reviewer/approval/pending_detections/
   - Approve detections: /api/alerts/reviewer/approval/approve_detection/
   - Reject detections: /api/alerts/reviewer/approval/reject_detection/

4. DIRECTORY STRUCTURE:
   - pending_detections/: Videos awaiting review
   - confirmed_detections/: Approved videos (with DB records)
   - rejected_detections/: Rejected videos (no DB records)

5. UPDATE YOUR FRONTEND:
   - Update reviewer dashboard to use new approval endpoints
   - Remove old pending alert workflows
   - Add detection video preview functionality

The system is now configured to only create database records
upon reviewer approval, as requested.
        """
        
        self.stdout.write(instructions)
        
        self.stdout.write(
            self.style.WARNING('\nIMPORTANT: Update your frontend code to use the new endpoints!')
        )
        self.stdout.write(
            self.style.SUCCESS('='*60 + '\n')
        )