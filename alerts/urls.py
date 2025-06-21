# alerts/urls.py - Updated with reviewer workflow endpoints

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import AlertViewSet
from .reviewer_views import ReviewerAlertViewSet, AlertReviewViewSet, ReviewerAssignmentViewSet

# Main router for regular alert views
router = DefaultRouter()
router.register(r'', AlertViewSet, basename='alert')

# Reviewer router for reviewer-specific endpoints
reviewer_router = DefaultRouter()
reviewer_router.register(r'pending', ReviewerAlertViewSet, basename='reviewer-alert')
reviewer_router.register(r'reviews', AlertReviewViewSet, basename='alert-review')
reviewer_router.register(r'assignments', ReviewerAssignmentViewSet, basename='reviewer-assignment')

urlpatterns = [
    # Regular alert endpoints (for end users)
    path('', include(router.urls)),
    
    # Reviewer workflow endpoints
    path('reviewer/', include(reviewer_router.urls)),
]

# Available endpoints:
# 
# Regular Alert Endpoints (End Users):
# GET    /api/alerts/                     - List user's alerts
# GET    /api/alerts/{id}/               - Get specific alert
# POST   /api/alerts/                    - Create alert (manual)
# PUT    /api/alerts/{id}/status/        - Update alert status
# GET    /api/alerts/summary/            - Get alert summary statistics
# GET    /api/alerts/{id}/video/         - Get alert video file
#
# Reviewer Workflow Endpoints:
# GET    /api/alerts/reviewer/pending/                    - List pending alerts for review
# GET    /api/alerts/reviewer/pending/{id}/              - Get specific pending alert
# POST   /api/alerts/reviewer/pending/{id}/confirm/      - Confirm alert as true positive
# POST   /api/alerts/reviewer/pending/{id}/dismiss/      - Dismiss alert as not actionable
# POST   /api/alerts/reviewer/pending/{id}/mark_false_positive/ - Mark as false positive
# GET    /api/alerts/reviewer/pending/pending_summary/   - Get reviewer dashboard summary
#
# GET    /api/alerts/reviewer/reviews/                   - List alert reviews
# GET    /api/alerts/reviewer/reviews/{id}/              - Get specific review
# POST   /api/alerts/reviewer/reviews/                   - Create review record
#
# GET    /api/alerts/reviewer/assignments/               - List reviewer assignments (Admin)
# POST   /api/alerts/reviewer/assignments/               - Create reviewer assignment (Admin)
# GET    /api/alerts/reviewer/assignments/{id}/          - Get specific assignment (Admin)
# PUT    /api/alerts/reviewer/assignments/{id}/          - Update assignment (Admin)
# DELETE /api/alerts/reviewer/assignments/{id}/          - Delete assignment (Admin)
# GET    /api/alerts/reviewer/assignments/available_reviewers/ - Get available reviewers (Admin)
# GET    /api/alerts/reviewer/assignments/reviewer_workload/   - Get reviewer workload stats (Admin)