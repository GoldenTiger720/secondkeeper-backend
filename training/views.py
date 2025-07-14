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