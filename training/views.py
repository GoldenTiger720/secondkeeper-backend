from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from alerts.models import Alert, AlertReview
from .serializers import AlertTrainingDataSerializer
from django.db.models import Prefetch


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_training_data(request):
    """
    Get all alert review data for training purposes.
    
    Query Parameters:
    - alert_type: Filter by alert type (fire_smoke, fall, violence, choking)
    """
    alert_type = request.query_params.get('alert_type', None)
    
    # Build query with optimized prefetch
    queryset = Alert.objects.select_related(
        'camera',
        'camera__user',
        'resolved_by',
        'reviewed_by'
    ).prefetch_related(
        Prefetch(
            'review_history',
            queryset=AlertReview.objects.select_related('reviewer')
        )
    )
    
    # Apply alert_type filter if provided
    if alert_type:
        if alert_type not in ['fire_smoke', 'fall', 'violence', 'choking']:
            return Response(
                {'error': 'Invalid alert_type. Must be one of: fire_smoke, fall, violence, choking'},
                status=status.HTTP_400_BAD_REQUEST
            )
        queryset = queryset.filter(alert_type=alert_type)
    
    # Order by detection time (newest first)
    queryset = queryset.order_by('-detection_time')
    
    # Serialize the data
    serializer = AlertTrainingDataSerializer(queryset, many=True)
    
    # Prepare response data
    response_data = {
        'count': queryset.count(),
        'alert_type': alert_type if alert_type else 'all',
        'data': serializer.data
    }
    
    # Add statistics
    stats = {
        'total_alerts': queryset.count(),
        'confirmed': queryset.filter(status='confirmed').count(),
        'dismissed': queryset.filter(status='dismissed').count(),
        'false_positive': queryset.filter(status='false_positive').count(),
        'pending_review': queryset.filter(status='pending_review').count()
    }
    response_data['statistics'] = stats
    
    return Response(response_data, status=status.HTTP_200_OK)