# cameras/urls.py - UPDATED to include enhanced features in existing endpoints

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import CameraViewSet
from .detection_views import (
    detection_service_status,
    start_detection_service,
    stop_detection_service,
    restart_detection_service,
    active_camera_processors,
    detection_statistics,
    # New enhanced features
    test_video_management,
    upload_test_video,
    delete_test_video
)

router = DefaultRouter()
router.register(r'', CameraViewSet, basename='camera')

urlpatterns = [
    path('', include(router.urls)),
    
    # Detection service control endpoints (now supports both standard and enhanced)
    path('detection/status/', detection_service_status, name='detection-status'),
    path('detection/start/', start_detection_service, name='detection-start'),
    path('detection/stop/', stop_detection_service, name='detection-stop'),
    path('detection/restart/', restart_detection_service, name='detection-restart'),
    path('detection/processors/', active_camera_processors, name='detection-processors'),
    path('detection/statistics/', detection_statistics, name='detection-statistics'),
    
    # Enhanced features (only work if enhanced detection is available)
    path('detection/test-videos/', test_video_management, name='test-video-management'),
    path('detection/test-videos/upload/', upload_test_video, name='upload-test-video'),
    path('detection/test-videos/delete/', delete_test_video, name='delete-test-video'),
]

"""
API ENDPOINTS SUMMARY:

EXISTING ENDPOINTS (Enhanced with video support when available):
- GET  /api/cameras/detection/status/           - Service status (shows video + camera info)
- POST /api/cameras/detection/start/            - Start service (enhanced if available)
- POST /api/cameras/detection/stop/             - Stop service
- POST /api/cameras/detection/restart/          - Restart service
- GET  /api/cameras/detection/processors/       - Active processors (video + camera)
- GET  /api/cameras/detection/statistics/       - Detection statistics

NEW ENDPOINTS (Only work with enhanced detection):
- GET  /api/cameras/detection/test-videos/      - List test videos
- POST /api/cameras/detection/test-videos/upload/ - Upload test video
- DELETE /api/cameras/detection/test-videos/delete/ - Delete test video

USAGE EXAMPLES:

1. Check if enhanced features are available:
   GET /api/cameras/detection/status/
   Response: {"enhanced_features_available": true/false}

2. Upload a test video:
   POST /api/cameras/detection/test-videos/upload/
   Form data: {"video_file": <file>}

3. List test videos:
   GET /api/cameras/detection/test-videos/

4. Delete a test video:
   DELETE /api/cameras/detection/test-videos/delete/
   JSON: {"filename": "test_video.mp4"}

The enhanced features will return 404 if enhanced detection is not available.
"""