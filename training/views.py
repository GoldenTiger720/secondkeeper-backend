from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from alerts.models import Alert, AlertReview
from .serializers import AlertReviewTrainingDataSerializer
from django.db.models import Prefetch, Count


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_training_data(request):
    """
    Get all alert review data for training purposes from the AlertReview table.
    
    Query Parameters:
    - alert_type: Filter by alert type (fire_smoke, fall, violence, choking)
    """
    alert_type = request.query_params.get('alert_type', None)
    
    # Build query from AlertReview table with optimized prefetch
    queryset = AlertReview.objects.select_related(
        'alert',
        'alert__camera',
        'alert__camera__user',
        'reviewer'
    )
    
    # Apply alert_type filter if provided
    if alert_type:
        if alert_type not in ['fire_smoke', 'fall', 'violence', 'choking']:
            return Response(
                {'error': 'Invalid alert_type. Must be one of: fire_smoke, fall, violence, choking'},
                status=status.HTTP_400_BAD_REQUEST
            )
        queryset = queryset.filter(alert__alert_type=alert_type)
    
    # Order by review time (newest first)
    queryset = queryset.order_by('-review_time')
    
    # Serialize the data
    serializer = AlertReviewTrainingDataSerializer(queryset, many=True)
    
    # Prepare response data
    response_data = {
        'count': queryset.count(),
        'alert_type': alert_type if alert_type else 'all',
        'data': serializer.data
    }
    
    # Add statistics based on review actions
    stats = {
        'total_reviews': queryset.count(),
        'confirmed': queryset.filter(action='confirmed').count(),
        'dismissed': queryset.filter(action='dismissed').count(),
        'false_positive': queryset.filter(action='false_positive').count(),
        'escalated': queryset.filter(action='escalated').count()
    }
    
    # Add alert type breakdown if no specific type was requested
    if not alert_type:
        alert_type_stats = queryset.values('alert__alert_type').annotate(
            count=Count('id')
        ).order_by('alert__alert_type')
        stats['by_alert_type'] = {
            item['alert__alert_type']: item['count'] 
            for item in alert_type_stats
        }
    
    response_data['statistics'] = stats
    
    return Response(response_data, status=status.HTTP_200_OK)