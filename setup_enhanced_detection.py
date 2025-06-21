# setup_enhanced_detection.py - Setup script for enhanced detection integration

import os
import django
from django.conf import settings

def setup_enhanced_detection():
    """
    Setup script to integrate enhanced detection system
    """
    
    print("Setting up Enhanced Detection System...")
    
    # 1. Create necessary directories
    directories = [
        os.path.join(settings.MEDIA_ROOT, 'testvideo'),
        os.path.join(settings.MEDIA_ROOT, 'test_detections'),
        os.path.join(settings.MEDIA_ROOT, 'test_detections', 'fire_smoke'),
        os.path.join(settings.MEDIA_ROOT, 'test_detections', 'fall'),
        os.path.join(settings.MEDIA_ROOT, 'test_detections', 'violence'),
        os.path.join(settings.MEDIA_ROOT, 'test_detections', 'choking'),
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # 2. Apply patches to existing classes
    try:
        # Apply patch to EnhancedVideoProcessor
        from utils.enhanced_video_processor_patch import add_test_detection_support
        from utils.enhanced_video_processor import EnhancedVideoProcessor
        add_test_detection_support(EnhancedVideoProcessor)
        print("Applied patch to EnhancedVideoProcessor")
    except ImportError as e:
        print(f"Warning: Could not apply EnhancedVideoProcessor patch: {e}")
    
    # 3. Update URL configuration
    try:
        update_urls()
        print("Updated URL configuration")
    except Exception as e:
        print(f"Warning: Could not update URLs automatically: {e}")
        print("Please manually add enhanced detection URLs to cameras/urls.py")
    
    # 4. Create sample configuration
    create_sample_config()
    
    print("\nEnhanced Detection System setup complete!")
    print("\nNext steps:")
    print("1. Update your cameras/urls.py to include enhanced detection URLs")
    print("2. Run: python manage.py migrate")
    print("3. Place test videos in media/testvideo/ directory")
    print("4. Start the service: python manage.py start_enhanced_detection")
    print("\nAPI Endpoints:")
    print("- GET  /api/cameras/detection/enhanced/status/")
    print("- POST /api/cameras/detection/enhanced/start/")
    print("- POST /api/cameras/detection/enhanced/stop/")
    print("- GET  /api/cameras/detection/enhanced/test-videos/")
    print("- POST /api/cameras/detection/enhanced/test-videos/upload/")

def update_urls():
    """Update URL configuration (if possible)"""
    cameras_urls_path = os.path.join('cameras', 'urls.py')
    
    if os.path.exists(cameras_urls_path):
        with open(cameras_urls_path, 'r') as f:
            content = f.read()
        
        # Check if enhanced URLs are already included
        if 'enhanced_detection_urls' not in content:
            # Add import and include
            enhanced_import = "from . import enhanced_detection_urls"
            enhanced_include = "    path('detection/', include('cameras.enhanced_detection_urls')),"
            
            # Find import section
            lines = content.split('\n')
            import_line = -1
            urlpatterns_line = -1
            
            for i, line in enumerate(lines):
                if line.strip().startswith('from') and 'import' in line:
                    import_line = i
                elif 'urlpatterns = [' in line:
                    urlpatterns_line = i
            
            if import_line >= 0:
                lines.insert(import_line + 1, enhanced_import)
            
            if urlpatterns_line >= 0:
                # Find the end of urlpatterns
                for i in range(urlpatterns_line + 1, len(lines)):
                    if ']' in lines[i] and 'urlpatterns' not in lines[i]:
                        lines.insert(i, enhanced_include)
                        break
            
            # Write back the file
            with open(cameras_urls_path, 'w') as f:
                f.write('\n'.join(lines))

def create_sample_config():
    """Create sample configuration file"""
    config_content = """
# Enhanced Detection System Configuration

## Directory Structure:
- media/testvideo/           # Place test video files here
- media/test_detections/     # Test detection results stored here
- media/detected_videos/     # Regular detection videos

## Supported Video Formats:
- .mp4, .avi, .mov, .mkv, .wmv, .flv, .webm

## Detection Priority:
1. Video files in media/testvideo/ (highest priority)
2. Camera streams (when no video files exist)

## Management Commands:
```bash
# Start enhanced detection service
python manage.py start_enhanced_detection

# Manage test videos
python manage.py manage_test_videos list
python manage.py manage_test_videos add --file /path/to/video.mp4
python manage.py manage_test_videos remove --file video.mp4
python manage.py manage_test_videos clear
python manage.py manage_test_videos stats
```

## API Usage Examples:

### Check service status:
GET /api/cameras/detection/enhanced/status/

### Upload test video:
POST /api/cameras/detection/enhanced/test-videos/upload/
Content-Type: multipart/form-data
{
    "video_file": <video file>
}

### Get test video list:
GET /api/cameras/detection/enhanced/test-videos/

### Delete test video:
DELETE /api/cameras/detection/enhanced/test-videos/delete/
{
    "filename": "test_video.mp4"
}

## Detection Flow:
1. System checks media/testvideo/ for video files
2. If videos exist: Processes video files with object detection
3. If no videos: Falls back to camera stream processing
4. All detections create alerts in "pending_review" status
5. Reviewers can approve/reject detections via reviewer API

## Notes:
- Test video detection creates TEST alerts in the database
- All detections go through reviewer workflow
- Video files are processed sequentially
- Camera processing resumes when all videos are processed
"""
    
    config_path = os.path.join(settings.BASE_DIR, 'ENHANCED_DETECTION_CONFIG.md')
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"Created configuration file: {config_path}")

if __name__ == '__main__':
    import sys
    import os
    
    # Add the project root to Python path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # Setup Django
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'security_ai_system.settings')
    django.setup()
    
    setup_enhanced_detection()

"""
MANUAL INTEGRATION STEPS:

1. Add the enhanced detection URLs to cameras/urls.py:

```python
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import CameraViewSet
from .detection_views import (
    detection_service_status,
    start_detection_service,
    stop_detection_service,
    restart_detection_service,
    active_camera_processors,
    detection_statistics
)

router = DefaultRouter()
router.register(r'', CameraViewSet, basename='camera')

urlpatterns = [
    path('', include(router.urls)),
    
    # Original detection service control endpoints
    path('detection/status/', detection_service_status, name='detection-status'),
    path('detection/start/', start_detection_service, name='detection-start'),
    path('detection/stop/', stop_detection_service, name='detection-stop'),
    path('detection/restart/', restart_detection_service, name='detection-restart'),
    path('detection/processors/', active_camera_processors, name='detection-processors'),
    path('detection/statistics/', detection_statistics, name='detection-statistics'),
    
    # Enhanced detection endpoints
    path('detection/', include('cameras.enhanced_detection_urls')),
]
```

2. Create the enhanced_detection_urls.py file in the cameras app

3. Create the enhanced_detection_views.py file in the cameras app

4. Add the management commands to cameras/management/commands/

5. Create the utils files for enhanced detection

6. Run the setup:
```bash
python setup_enhanced_detection.py
python manage.py migrate
```

7. Start the enhanced detection service:
```bash
python manage.py start_enhanced_detection
```

8. Test by placing a video file in media/testvideo/ and checking the status:
```bash
curl http://localhost:8000/api/cameras/detection/enhanced/status/
```
"""