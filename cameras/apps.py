# cameras/apps.py - Updated to use enhanced detection manager properly

from django.apps import AppConfig
import threading
import time
import logging

logger = logging.getLogger('security_ai')

class CamerasConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'cameras'
    
    def ready(self):
        """Called when Django is ready - now uses enhanced detection manager with proper alert storage"""
        # Only start detection service in the main process
        # and not during migrations or management commands
        import os
        import sys
        
        # Check if this is the main Django process
        if (os.environ.get('RUN_MAIN') == 'true' or 
            'runserver' not in sys.argv and 
            'migrate' not in sys.argv and
            'makemigrations' not in sys.argv and
            'collectstatic' not in sys.argv and
            'shell' not in sys.argv):
            
            # Start ENHANCED detection service after a short delay
            def start_enhanced_detection_service():
                time.sleep(5)  # Wait for Django to fully initialize
                try:
                    # Import the ENHANCED detection manager
                    from utils.enhanced_detection_manager import enhanced_detection_manager
                    logger.info("Auto-starting Enhanced Camera Detection Service with Alert Storage...")
                    enhanced_detection_manager.start()
                    logger.info("Enhanced Camera Detection Service started automatically with proper alert storage and 10-second video recording")
                except Exception as e:
                    logger.error(f"Failed to auto-start enhanced detection service: {str(e)}")
                    # Fallback to original detection manager
                    try:
                        from utils.camera_detection_manager import detection_manager
                        logger.info("Falling back to original detection manager...")
                        detection_manager.start()
                        logger.info("Original Camera Detection Service started as fallback")
                    except Exception as e2:
                        logger.error(f"Failed to start fallback detection service: {str(e2)}")
            
            # Start in a separate thread
            thread = threading.Thread(target=start_enhanced_detection_service, daemon=True)
            thread.start()