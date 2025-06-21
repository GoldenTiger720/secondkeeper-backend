# utils/__init__.py - Updated to use enhanced detection manager by default

from .model_manager import ModelManager
from .enhanced_video_processor import EnhancedVideoProcessor
from .permissions import IsOwnerOrAdmin, IsAdminUser, IsManagerOrAdminOrReviewer
from .exception_handlers import custom_exception_handler
from .stream_proxy import StreamProxy

# Import enhanced detection manager as the default
try:
    from .enhanced_detection_manager import enhanced_detection_manager as detection_manager
    ENHANCED_DETECTION_AVAILABLE = True
except ImportError:
    # Fallback to original if enhanced is not available
    from .camera_detection_manager import detection_manager
    ENHANCED_DETECTION_AVAILABLE = False

__all__ = [
    'ModelManager', 
    'EnhancedVideoProcessor', 
    'IsOwnerOrAdmin',
    'IsAdminUser',
    'IsManagerOrAdminOrReviewer',
    'custom_exception_handler',
    'StreamProxy',
    'detection_manager',
    'ENHANCED_DETECTION_AVAILABLE'
]


# cameras/management/commands/start_detection.py - Updated to use enhanced detection

import signal
import sys
import time
import logging
from django.core.management.base import BaseCommand
from django.conf import settings

logger = logging.getLogger('security_ai')

class Command(BaseCommand):
    help = 'Start the automatic camera detection service (enhanced version with video file support)'
    
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
            '--force-standard',
            action='store_true',
            help='Force use of standard detection (cameras only)',
        )
        
    def handle(self, *args, **options):
        # Try to use enhanced detection manager first
        if not options.get('force_standard'):
            try:
                from utils.enhanced_detection_manager import enhanced_detection_manager
                self.detection_manager = enhanced_detection_manager
                self.stdout.write(self.style.SUCCESS('Starting Enhanced Camera Detection Service...'))
                self.stdout.write(self.style.WARNING('This service will prioritize video files over camera streams.'))
                
                # Create test video directory
                import os
                test_video_dir = os.path.join(settings.MEDIA_ROOT, 'testvideo')
                os.makedirs(test_video_dir, exist_ok=True)
                self.stdout.write(f'Test video directory: {test_video_dir}')
                
            except ImportError:
                self.stdout.write(self.style.WARNING('Enhanced detection not available, using standard detection...'))
                from utils.camera_detection_manager import detection_manager
                self.detection_manager = detection_manager
        else:
            self.stdout.write(self.style.WARNING('Forced standard detection mode...'))
            from utils.camera_detection_manager import detection_manager
            self.detection_manager = detection_manager
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        try:
            # Start the detection manager
            self.detection_manager.start()
            
            if hasattr(self.detection_manager, 'active_video_processors'):
                self.stdout.write(
                    self.style.SUCCESS(
                        'Enhanced Camera Detection Service started successfully!\n'
                        'The service will automatically:\n'
                        '1. Process video files in media/testvideo/ (if any exist)\n'
                        '2. Monitor camera streams (when no video files exist)\n'
                        '3. Switch between modes automatically\n'
                        'Press Ctrl+C to stop the service.'
                    )
                )
            else:
                self.stdout.write(
                    self.style.SUCCESS(
                        'Standard Camera Detection Service started successfully!\n'
                        'The service is now monitoring all online cameras for threats.\n'
                        'Press Ctrl+C to stop the service.'
                    )
                )
            
            # Keep the service running
            while not self.shutdown_requested:
                # Show status for enhanced detection
                if hasattr(self.detection_manager, 'active_video_processors'):
                    video_count = len(self.detection_manager.active_video_processors)
                    camera_count = len(self.detection_manager.active_cameras)
                    
                    if video_count > 0:
                        self.stdout.write(f'Processing {video_count} video file(s)...')
                    elif camera_count > 0:
                        self.stdout.write(f'Monitoring {camera_count} camera(s)...')
                    else:
                        self.stdout.write('Waiting for video files or cameras...')
                
                time.sleep(10)
                
        except KeyboardInterrupt:
            self.stdout.write(self.style.WARNING('\nReceived interrupt signal...'))
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error in detection service: {str(e)}')
            )
            logger.error(f"Detection service error: {str(e)}")
        finally:
            self.shutdown()
            
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.stdout.write(self.style.WARNING(f'\nReceived signal {signum}. Shutting down...'))
        self.shutdown_requested = True
        
    def shutdown(self):
        """Shutdown the detection service gracefully"""
        try:
            self.stdout.write(self.style.WARNING('Stopping Detection Service...'))
            self.detection_manager.stop()
            self.stdout.write(self.style.SUCCESS('Detection Service stopped successfully.'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error during shutdown: {str(e)}'))
            logger.error(f"Shutdown error: {str(e)}")
            
        sys.exit(0)


# Simple replacement instruction for immediate usage:

"""
QUICK INTEGRATION STEPS:

1. Replace the import in cameras/detection_views.py:

   OLD:
   from utils.camera_detection_manager import detection_manager

   NEW:
   try:
       from utils.enhanced_detection_manager import enhanced_detection_manager as detection_manager
   except ImportError:
       from utils.camera_detection_manager import detection_manager

2. Replace the import in cameras/apps.py:

   OLD:
   from utils.camera_detection_manager import detection_manager

   NEW:
   try:
       from utils.enhanced_detection_manager import enhanced_detection_manager as detection_manager
   except ImportError:
       from utils.camera_detection_manager import detection_manager

3. Create the media/testvideo directory:
   mkdir -p media/testvideo

4. Place any test videos in media/testvideo/

5. Start the service:
   python manage.py start_detection

The system will now:
- Check media/testvideo/ for video files first
- Process videos if found (creates TEST alerts)
- Fall back to camera streams if no videos
- Automatically switch between modes

No other changes needed!
"""