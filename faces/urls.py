from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import AuthorizedFaceViewSet, FaceUploadView

router = DefaultRouter()
router.register(r'', AuthorizedFaceViewSet, basename='face')

urlpatterns = [
    path('', include(router.urls)),
    path('upload_face_image/', FaceUploadView.as_view(), name='face-upload'),
]