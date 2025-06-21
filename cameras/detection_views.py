# cameras/detection_views.py - UPDATED to include enhanced features

from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from utils.permissions import IsAdminUser
from cameras.models import Camera
from alerts.models import Alert
from django.db.models import Count, Q
from django.utils import timezone
from datetime import timedelta
import os
import glob
import logging
from django.conf import settings

logger = logging.getLogger('security_ai')

# Import BOTH detection managers with fallback
try:
    from utils.enhanced_detection_manager import enhanced_detection_manager
    ENHANCED_AVAILABLE = True
    logger.info("Enhanced detection manager available")
except ImportError:
    ENHANCED_AVAILABLE = False
    logger.warning("Enhanced detection manager not available")

from utils.camera_detection_manager import detection_manager as original_detection_manager

def get_active_detection_manager():
    """Get the appropriate detection manager"""
    if ENHANCED_AVAILABLE:
        return enhanced_detection_manager
    else:
        return original_detection_manager

@api_view(['GET'])
@permission_classes([IsAuthenticated, IsAdminUser])
def detection_service_status(request):
    """Get the status of the detection service (enhanced or original)"""
    try:
        detection_manager = get_active_detection_manager()
        
        # Get service status
        is_running = detection_manager.is_running
        
        # Enhanced detection manager has additional properties
        if ENHANCED_AVAILABLE and hasattr(detection_manager, 'active_video_processors'):
            active_video_processors = len(detection_manager.active_video_processors)
            active_cameras_count = len(detection_manager.active_cameras)
            
            # Check for test videos
            test_video_dir = os.path.join(settings.MEDIA_ROOT, 'testvideo')
            video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv', '*.webm']
            test_videos = []
            
            for extension in video_extensions:
                pattern = os.path.join(test_video_dir, extension)
                test_videos.extend(glob.glob(pattern))
            
            test_video_info = []
            for video_file in test_videos:
                file_size = os.path.getsize(video_file)
                test_video_info.append({
                    'name': os.path.basename(video_file),
                    'size_mb': round(file_size / (1024 * 1024), 2)
                })
            
            current_mode = 'enhanced_detection'
            if test_videos:
                processing_mode = 'video_file_processing'
                source_info = f"{len(test_videos)} test video file(s)"
            elif active_cameras_count > 0:
                processing_mode = 'camera_streaming'
                source_info = f"{active_cameras_count} camera stream(s)"
            else:
                processing_mode = 'idle'
                source_info = "No active sources"
                
        else:
            # Original detection manager
            active_cameras_count = len(detection_manager.active_cameras)
            active_video_processors = 0
            current_mode = 'standard_detection'
            processing_mode = 'camera_streaming' if active_cameras_count > 0 else 'idle'
            source_info = f"{active_cameras_count} camera stream(s)" if active_cameras_count > 0 else "No active cameras"
            test_videos = []
            test_video_info = []
        
        # Get camera statistics
        total_cameras = Camera.objects.count()
        online_cameras = Camera.objects.filter(status='online').count()
        detection_enabled_cameras = Camera.objects.filter(
            status='online', 
            detection_enabled=True
        ).count()
        
        # Get recent alerts (last 24 hours)
        yesterday = timezone.now() - timedelta(days=1)
        recent_alerts = Alert.objects.filter(
            detection_time__gte=yesterday
        ).count()
        
        # Get test alerts if enhanced mode
        test_alerts = 0
        if ENHANCED_AVAILABLE:
            test_alerts = Alert.objects.filter(
                title__icontains='TEST',
                detection_time__gte=yesterday
            ).count()
        
        # Get alerts by type (last 24 hours)
        alerts_by_type = Alert.objects.filter(
            detection_time__gte=yesterday
        ).values('alert_type').annotate(count=Count('id'))
        
        response_data = {
            'service_running': is_running,
            'detection_mode': current_mode,
            'processing_mode': processing_mode,
            'source_info': source_info,
            'enhanced_features_available': ENHANCED_AVAILABLE,
            'processors': {
                'video_files': active_video_processors,
                'cameras': active_cameras_count
            },
            'cameras': {
                'total': total_cameras,
                'online': online_cameras,
                'detection_enabled': detection_enabled_cameras
            },
            'alerts_24h': {
                'total': recent_alerts,
                'test_alerts': test_alerts,
                'by_type': {
                    item['alert_type']: item['count'] 
                    for item in alerts_by_type
                }
            },
            'last_update': timezone.now().isoformat()
        }
        
        # Add enhanced features data if available
        if ENHANCED_AVAILABLE:
            response_data['test_videos'] = {
                'count': len(test_videos),
                'files': test_video_info,
                'directory': os.path.join(settings.MEDIA_ROOT, 'testvideo')
            }
        
        return Response({
            'success': True,
            'data': response_data,
            'message': f'Detection service status retrieved successfully ({current_mode}).',
            'errors': []
        })
        
    except Exception as e:
        logger.error(f"Error getting detection service status: {str(e)}")
        return Response({
            'success': False,
            'data': {},
            'message': 'Error retrieving detection service status.',
            'errors': [str(e)]
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
@permission_classes([IsAuthenticated, IsAdminUser])
def start_detection_service(request):
    """Start the detection service (enhanced or original)"""
    try:
        detection_manager = get_active_detection_manager()
        
        if detection_manager.is_running:
            return Response({
                'success': True,
                'data': {
                    'status': 'already_running',
                    'mode': 'enhanced' if ENHANCED_AVAILABLE else 'standard'
                },
                'message': f'Detection service is already running ({"enhanced" if ENHANCED_AVAILABLE else "standard"} mode).',
                'errors': []
            })
        
        detection_manager.start()
        
        return Response({
            'success': True,
            'data': {
                'status': 'started',
                'mode': 'enhanced' if ENHANCED_AVAILABLE else 'standard'
            },
            'message': f'Detection service started successfully ({"enhanced" if ENHANCED_AVAILABLE else "standard"} mode).',
            'errors': []
        })
        
    except Exception as e:
        logger.error(f"Error starting detection service: {str(e)}")
        return Response({
            'success': False,
            'data': {},
            'message': 'Error starting detection service.',
            'errors': [str(e)]
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
@permission_classes([IsAuthenticated, IsAdminUser])
def stop_detection_service(request):
    """Stop the detection service (enhanced or original)"""
    try:
        detection_manager = get_active_detection_manager()
        
        if not detection_manager.is_running:
            return Response({
                'success': True,
                'data': {
                    'status': 'already_stopped'
                },
                'message': 'Detection service is already stopped.',
                'errors': []
            })
        
        detection_manager.stop()
        
        return Response({
            'success': True,
            'data': {
                'status': 'stopped'
            },
            'message': 'Detection service stopped successfully.',
            'errors': []
        })
        
    except Exception as e:
        logger.error(f"Error stopping detection service: {str(e)}")
        return Response({
            'success': False,
            'data': {},
            'message': 'Error stopping detection service.',
            'errors': [str(e)]
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
@permission_classes([IsAuthenticated, IsAdminUser])
def restart_detection_service(request):
    """Restart the detection service (enhanced or original)"""
    try:
        detection_manager = get_active_detection_manager()
        
        # Stop if running
        if detection_manager.is_running:
            detection_manager.stop()
            
        # Start again
        detection_manager.start()
        
        return Response({
            'success': True,
            'data': {'status': 'restarted'},
            'message': 'Detection service restarted successfully.',
            'errors': []
        })
        
    except Exception as e:
        logger.error(f"Error restarting detection service: {str(e)}")
        return Response({
            'success': False,
            'data': {},
            'message': 'Error restarting detection service.',
            'errors': [str(e)]
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
@permission_classes([IsAuthenticated, IsAdminUser])
def active_camera_processors(request):
    """Get list of active camera processors"""
    try:
        detection_manager = get_active_detection_manager()
        active_processors = []
        
        for camera_id, processor in detection_manager.active_cameras.items():
            processor_info = {
                'camera_id': camera_id,
                'camera_name': processor.camera.name,
                'is_running': processor.is_running,
                'frame_count': processor.frame_count,
                'confidence_threshold': processor.confidence_threshold,
                'detectors_enabled': {
                    'fire_smoke': processor.camera.fire_smoke_detection,
                    'fall': processor.camera.fall_detection,
                    'violence': processor.camera.violence_detection,
                    'choking': processor.camera.choking_detection
                }
            }
            active_processors.append(processor_info)
        
        # Add video processor info if enhanced mode
        video_processors = []
        if ENHANCED_AVAILABLE and hasattr(detection_manager, 'active_video_processors'):
            for video_file, processor in detection_manager.active_video_processors.items():
                video_processors.append({
                    'video_file': os.path.basename(video_file),
                    'video_path': video_file,
                    'is_running': processor.is_running,
                    'frame_count': processor.frame_count
                })
        
        return Response({
            'success': True,
            'data': {
                'detection_mode': 'enhanced' if ENHANCED_AVAILABLE else 'standard',
                'camera_processors': active_processors,
                'video_processors': video_processors,
                'total_camera_count': len(active_processors),
                'total_video_count': len(video_processors)
            },
            'message': 'Active processors retrieved successfully.',
            'errors': []
        })
        
    except Exception as e:
        logger.error(f"Error getting active processors: {str(e)}")
        return Response({
            'success': False,
            'data': {},
            'message': 'Error retrieving active processors.',
            'errors': [str(e)]
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
@permission_classes([IsAuthenticated, IsAdminUser])
def detection_statistics(request):
    """Get detection statistics"""
    try:
        # Get time range from query params
        days = int(request.GET.get('days', 7))
        start_date = timezone.now() - timedelta(days=days)
        
        # Total alerts in time range
        total_alerts = Alert.objects.filter(
            detection_time__gte=start_date
        ).count()
        
        # Test alerts if enhanced mode
        test_alerts = 0
        if ENHANCED_AVAILABLE:
            test_alerts = Alert.objects.filter(
                title__icontains='TEST',
                detection_time__gte=start_date
            ).count()
        
        # Alerts by severity
        alerts_by_severity = Alert.objects.filter(
            detection_time__gte=start_date
        ).values('severity').annotate(count=Count('id'))
        
        # Alerts by camera
        alerts_by_camera = Alert.objects.filter(
            detection_time__gte=start_date
        ).select_related('camera').values(
            'camera__id', 'camera__name'
        ).annotate(count=Count('id')).order_by('-count')[:10]
        
        # Alerts by status
        alerts_by_status = Alert.objects.filter(
            detection_time__gte=start_date
        ).values('status').annotate(count=Count('id'))
        
        # Daily alert counts
        daily_alerts = []
        for i in range(days):
            day_start = timezone.now().replace(
                hour=0, minute=0, second=0, microsecond=0
            ) - timedelta(days=i)
            day_end = day_start + timedelta(days=1)
            
            day_count = Alert.objects.filter(
                detection_time__gte=day_start,
                detection_time__lt=day_end
            ).count()
            
            daily_alerts.append({
                'date': day_start.date().isoformat(),
                'count': day_count
            })
        
        daily_alerts.reverse()  # Oldest first
        
        response_data = {
            'time_range_days': days,
            'total_alerts': total_alerts,
            'test_alerts': test_alerts,
            'enhanced_mode': ENHANCED_AVAILABLE,
            'alerts_by_severity': {
                item['severity']: item['count']
                for item in alerts_by_severity
            },
            'alerts_by_camera': [
                {
                    'camera_id': item['camera__id'],
                    'camera_name': item['camera__name'],
                    'count': item['count']
                }
                for item in alerts_by_camera
            ],
            'alerts_by_status': {
                item['status']: item['count']
                for item in alerts_by_status
            },
            'daily_alerts': daily_alerts
        }
        
        return Response({
            'success': True,
            'data': response_data,
            'message': 'Detection statistics retrieved successfully.',
            'errors': []
        })
        
    except Exception as e:
        logger.error(f"Error getting detection statistics: {str(e)}")
        return Response({
            'success': False,
            'data': {},
            'message': 'Error retrieving detection statistics.',
            'errors': [str(e)]
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# NEW ENHANCED FEATURES - Only available if enhanced detection is enabled

@api_view(['GET'])
@permission_classes([IsAuthenticated, IsAdminUser])
def test_video_management(request):
    """Get test video management information (Enhanced mode only)"""
    if not ENHANCED_AVAILABLE:
        return Response({
            'success': False,
            'data': {},
            'message': 'Enhanced detection features not available.',
            'errors': ['Enhanced detection manager not installed']
        }, status=status.HTTP_404_NOT_FOUND)
    
    try:
        test_video_dir = os.path.join(settings.MEDIA_ROOT, 'testvideo')
        os.makedirs(test_video_dir, exist_ok=True)
        
        # Get all video files
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv', '*.webm']
        video_files = []
        
        for extension in video_extensions:
            pattern = os.path.join(test_video_dir, extension)
            video_files.extend(glob.glob(pattern))
        
        video_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        video_info = []
        total_size = 0
        
        for video_file in video_files:
            file_size = os.path.getsize(video_file)
            total_size += file_size
            
            video_info.append({
                'name': os.path.basename(video_file),
                'size_mb': round(file_size / (1024 * 1024), 2),
                'modified_time': timezone.datetime.fromtimestamp(
                    os.path.getmtime(video_file)
                ).isoformat()
            })
        
        detection_manager = get_active_detection_manager()
        is_processing_videos = len(detection_manager.active_video_processors) > 0
        
        return Response({
            'success': True,
            'data': {
                'directory': test_video_dir,
                'video_files': video_info,
                'total_files': len(video_files),
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'is_processing': is_processing_videos,
                'supported_formats': ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'],
                'priority_info': {
                    'description': 'Video files have priority over camera streams',
                    'current_mode': 'video_processing' if video_files else 'camera_streaming'
                }
            },
            'message': 'Test video information retrieved successfully.',
            'errors': []
        })
        
    except Exception as e:
        logger.error(f"Error getting test video information: {str(e)}")
        return Response({
            'success': False,
            'data': {},
            'message': 'Error retrieving test video information.',
            'errors': [str(e)]
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
@permission_classes([IsAuthenticated, IsAdminUser])
def upload_test_video(request):
    """Upload a test video file (Enhanced mode only)"""
    if not ENHANCED_AVAILABLE:
        return Response({
            'success': False,
            'data': {},
            'message': 'Enhanced detection features not available.',
            'errors': ['Enhanced detection manager not installed']
        }, status=status.HTTP_404_NOT_FOUND)
    
    try:
        if 'video_file' not in request.FILES:
            return Response({
                'success': False,
                'data': {},
                'message': 'No video file provided.',
                'errors': ['video_file is required']
            }, status=status.HTTP_400_BAD_REQUEST)
        
        video_file = request.FILES['video_file']
        
        # Validate file extension
        allowed_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
        file_extension = os.path.splitext(video_file.name)[1].lower()
        
        if file_extension not in allowed_extensions:
            return Response({
                'success': False,
                'data': {},
                'message': f'Unsupported file format: {file_extension}',
                'errors': [f'Allowed formats: {", ".join(allowed_extensions)}']
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Validate file size (max 500MB)
        max_size = 500 * 1024 * 1024  # 500MB
        if video_file.size > max_size:
            return Response({
                'success': False,
                'data': {},
                'message': 'File too large.',
                'errors': [f'Maximum file size is 500MB, uploaded file is {video_file.size / (1024*1024):.1f}MB']
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Save file to test directory
        test_video_dir = os.path.join(settings.MEDIA_ROOT, 'testvideo')
        os.makedirs(test_video_dir, exist_ok=True)
        
        file_path = os.path.join(test_video_dir, video_file.name)
        
        # Check if file already exists
        if os.path.exists(file_path):
            # Add timestamp to make it unique
            name, ext = os.path.splitext(video_file.name)
            timestamp = timezone.now().strftime('%Y%m%d_%H%M%S')
            file_path = os.path.join(test_video_dir, f"{name}_{timestamp}{ext}")
        
        # Save the file
        with open(file_path, 'wb+') as destination:
            for chunk in video_file.chunks():
                destination.write(chunk)
        
        file_size_mb = round(video_file.size / (1024 * 1024), 2)
        
        return Response({
            'success': True,
            'data': {
                'filename': os.path.basename(file_path),
                'size_mb': file_size_mb,
                'path': file_path,
                'message': 'Video uploaded successfully and will be processed with priority over camera streams.'
            },
            'message': 'Test video uploaded successfully.',
            'errors': []
        })
        
    except Exception as e:
        logger.error(f"Error uploading test video: {str(e)}")
        return Response({
            'success': False,
            'data': {},
            'message': 'Error uploading test video.',
            'errors': [str(e)]
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['DELETE'])
@permission_classes([IsAuthenticated, IsAdminUser])
def delete_test_video(request):
    """Delete a test video file (Enhanced mode only)"""
    if not ENHANCED_AVAILABLE:
        return Response({
            'success': False,
            'data': {},
            'message': 'Enhanced detection features not available.',
            'errors': ['Enhanced detection manager not installed']
        }, status=status.HTTP_404_NOT_FOUND)
    
    try:
        filename = request.data.get('filename')
        if not filename:
            return Response({
                'success': False,
                'data': {},
                'message': 'Filename is required.',
                'errors': ['filename parameter is required']
            }, status=status.HTTP_400_BAD_REQUEST)
        
        test_video_dir = os.path.join(settings.MEDIA_ROOT, 'testvideo')
        file_path = os.path.join(test_video_dir, filename)
        
        if not os.path.exists(file_path):
            return Response({
                'success': False,
                'data': {},
                'message': 'File not found.',
                'errors': [f'File {filename} not found in test directory']
            }, status=status.HTTP_404_NOT_FOUND)
        
        # Delete the file
        os.remove(file_path)
        
        # Check if any videos remain
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv', '*.webm']
        remaining_videos = []
        for extension in video_extensions:
            pattern = os.path.join(test_video_dir, extension)
            remaining_videos.extend(glob.glob(pattern))
        
        return Response({
            'success': True,
            'data': {
                'deleted_file': filename,
                'remaining_videos': len(remaining_videos),
                'mode_switch': len(remaining_videos) == 0
            },
            'message': f'Test video {filename} deleted successfully.' + 
                      (' System will switch to camera detection.' if len(remaining_videos) == 0 else ''),
            'errors': []
        })
        
    except Exception as e:
        logger.error(f"Error deleting test video: {str(e)}")
        return Response({
            'success': False,
            'data': {},
            'message': 'Error deleting test video.',
            'errors': [str(e)]
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)