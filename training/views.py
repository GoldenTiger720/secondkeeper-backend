from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from alerts.models import Alert, AlertReview
from .serializers import AlertReviewTrainingDataSerializer, TrainingFireSerializer, TrainingChokingSerializer, TrainingFallSerializer, TrainingViolenceSerializer
from .models import TrainingFire, TrainingChoking, TrainingFall, TrainingViolence
from django.db.models import Prefetch, Count


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_training_data(request):
    """
    Get training data from training tables based on alert_type.
    
    Query Parameters:
    - alert_type: Filter by alert type (fire_smoke, fall, violence, choking)
    """
    alert_type = request.query_params.get('alert_type', None)
    
    if not alert_type:
        return Response({
            'success': False,
            'data': {},
            'message': 'alert_type parameter is required.',
            'errors': ['Missing alert_type parameter.']
        }, status=status.HTTP_400_BAD_REQUEST)
    
    # Validate alert_type
    valid_alert_types = ['fire_smoke', 'fall', 'choking', 'violence']
    if alert_type not in valid_alert_types:
        return Response({
            'success': False,
            'data': {},
            'message': f'Invalid alert_type. Must be one of: {", ".join(valid_alert_types)}',
            'errors': ['Invalid alert_type parameter.']
        }, status=status.HTTP_400_BAD_REQUEST)
    
    # Get data from appropriate training table
    try:
        if alert_type == 'fire_smoke':
            queryset = TrainingFire.objects.all()
            serializer = TrainingFireSerializer(queryset, many=True)
        elif alert_type == 'fall':
            queryset = TrainingFall.objects.all()
            serializer = TrainingFallSerializer(queryset, many=True)
        elif alert_type == 'choking':
            queryset = TrainingChoking.objects.all()
            serializer = TrainingChokingSerializer(queryset, many=True)
        elif alert_type == 'violence':
            queryset = TrainingViolence.objects.all()
            serializer = TrainingViolenceSerializer(queryset, many=True)
        
        return Response({
            'success': True,
            'data': serializer.data,
            'message': f'Training data for {alert_type} retrieved successfully.',
            'errors': []
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response({
            'success': False,
            'data': {},
            'message': 'Error retrieving training data.',
            'errors': [str(e)]
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def delete_training_data(request):
    """
    Delete training data from training tables based on alert_type and alert_ids.
    
    Body Parameters:
    - alert_ids: List of alert IDs to delete
    - alert_type: Type of alert (fire_smoke, fall, violence, choking)
    """
    # Debug logging to see what data is being sent
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Request data: {request.data}")
    logger.info(f"Request method: {request.method}")
    logger.info(f"Request content type: {request.content_type}")
    
    alert_ids = request.data.get('alert_ids', [])
    alert_type = request.data.get('alert_type', None)
    
    logger.info(f"Parsed alert_ids: {alert_ids}")
    logger.info(f"Parsed alert_type: {alert_type}")
    
    # Validate input
    if not alert_ids:
        return Response({
            'success': False,
            'data': {},
            'message': 'alert_ids parameter is required.',
            'errors': ['Missing alert_ids parameter.']
        }, status=status.HTTP_400_BAD_REQUEST)
    
    if not alert_type:
        return Response({
            'success': False,
            'data': {},
            'message': 'alert_type parameter is required.',
            'errors': ['Missing alert_type parameter.']
        }, status=status.HTTP_400_BAD_REQUEST)
    
    # Validate alert_type
    valid_alert_types = ['fire_smoke', 'fall', 'choking', 'violence']
    if alert_type not in valid_alert_types:
        return Response({
            'success': False,
            'data': {},
            'message': f'Invalid alert_type. Must be one of: {", ".join(valid_alert_types)}',
            'errors': ['Invalid alert_type parameter.']
        }, status=status.HTTP_400_BAD_REQUEST)
    
    # Validate alert_ids format
    if not isinstance(alert_ids, list):
        return Response({
            'success': False,
            'data': {},
            'message': 'alert_ids must be an array.',
            'errors': ['alert_ids must be a list of integers.']
        }, status=status.HTTP_400_BAD_REQUEST)
    
    # Validate that all IDs are integers
    try:
        alert_ids = [int(alert_id) for alert_id in alert_ids]
    except (ValueError, TypeError):
        return Response({
            'success': False,
            'data': {},
            'message': 'Invalid alert ID format.',
            'errors': ['All alert IDs must be valid integers.']
        }, status=status.HTTP_400_BAD_REQUEST)
    
    # Delete data from appropriate training table
    try:
        deleted_count = 0
        
        if alert_type == 'fire_smoke':
            deleted_count = TrainingFire.objects.filter(id__in=alert_ids).count()
            TrainingFire.objects.filter(id__in=alert_ids).delete()
        elif alert_type == 'fall':
            deleted_count = TrainingFall.objects.filter(id__in=alert_ids).count()
            TrainingFall.objects.filter(id__in=alert_ids).delete()
        elif alert_type == 'choking':
            deleted_count = TrainingChoking.objects.filter(id__in=alert_ids).count()
            TrainingChoking.objects.filter(id__in=alert_ids).delete()
        elif alert_type == 'violence':
            deleted_count = TrainingViolence.objects.filter(id__in=alert_ids).count()
            TrainingViolence.objects.filter(id__in=alert_ids).delete()
        
        return Response({
            'success': True,
            'data': {
                'deleted_count': deleted_count,
                'alert_type': alert_type,
                'deleted_ids': alert_ids
            },
            'message': f'Successfully deleted {deleted_count} training records for {alert_type}.',
            'errors': []
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response({
            'success': False,
            'data': {},
            'message': 'Error deleting training data.',
            'errors': [str(e)]
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)