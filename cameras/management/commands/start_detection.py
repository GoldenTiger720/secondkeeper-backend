# cameras/management/commands/start_detection.py

import signal
import sys
import time
import logging
import os
from django.core.management.base import BaseCommand
from django.conf import settings
from utils.enhanced_detection_manager import enhanced_detection_manager

logger = logging.getLogger('security_ai')

class Command(BaseCommand):
    help = 'Start the enhanced detection service with video file priority'
    
    def __init__(self):
        super().__init__()
        self.shutdown_requested = False
        
    def add_arguments(self, parser):
        parser.add_argument(
            '--no-daemon',
            action='store_true',
            help='Run in foreground instead of daemon mode',
        )
        parser.add_argument(
            '--check-interval',
            type=int,
            default=5,
            help='Interval in seconds to check for video files (default: 5)',
        )
        
    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Starting Enhanced Detection Service...'))
        
        # Create test video directory if it doesn't exist
        test_video_dir = os.path.join(settings.MEDIA_ROOT, 'testvideo')
        os.makedirs(test_video_dir, exist_ok=True)
        
        self.stdout.write(
            self.style.WARNING(f'Test video directory: {test_video_dir}')
        )
        self.stdout.write(
            self.style.WARNING('Place video files in this directory for automatic detection')
        )
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        try:
            # Start the enhanced detection manager
            enhanced_detection_manager.start()
            
            self.stdout.write(
                self.style.SUCCESS(
                    'Enhanced Detection Service started successfully!\n'
                    'Priority order:\n'
                    '1. Video files in media/testvideo/ (if any exist)\n'
                    '2. Camera streams (if no video files)\n'
                    'Press Ctrl+C to stop the service.'
                )
            )
            
            # Keep the service running
            while not self.shutdown_requested:
                # Show current status
                if enhanced_detection_manager.active_video_processors:
                    video_count = len(enhanced_detection_manager.active_video_processors)
                    self.stdout.write(
                        self.style.WARNING(f'Processing {video_count} video file(s)...')
                    )
                elif enhanced_detection_manager.active_cameras:
                    camera_count = len(enhanced_detection_manager.active_cameras)
                    self.stdout.write(
                        self.style.WARNING(f'Processing {camera_count} camera stream(s)...')
                    )
                else:
                    self.stdout.write(
                        self.style.WARNING('No active video files or cameras...')
                    )
                
                time.sleep(options['check_interval'])
                
        except KeyboardInterrupt:
            self.stdout.write(self.style.WARNING('\nReceived interrupt signal...'))
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error in enhanced detection service: {str(e)}')
            )
            logger.error(f"Enhanced detection service error: {str(e)}")
        finally:
            self.shutdown()
            
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.stdout.write(self.style.WARNING(f'\nReceived signal {signum}. Shutting down...'))
        self.shutdown_requested = True
        
    def shutdown(self):
        """Shutdown the enhanced detection service gracefully"""
        try:
            self.stdout.write(self.style.WARNING('Stopping Enhanced Detection Service...'))
            enhanced_detection_manager.stop()
            self.stdout.write(self.style.SUCCESS('Enhanced Detection Service stopped successfully.'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error during shutdown: {str(e)}'))
            logger.error(f"Shutdown error: {str(e)}")
            
        sys.exit(0)