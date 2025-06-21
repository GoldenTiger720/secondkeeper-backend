# admin_panel/urls.py - Updated with all viewsets

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    UserAdminViewSet,
    CameraAdminViewSet,
    SystemStatusViewSet,
    SubscriptionViewSet,
    SystemSettingViewSet
)

router = DefaultRouter()
router.register(r'users', UserAdminViewSet, basename='admin-user')
router.register(r'cameras', CameraAdminViewSet, basename='admin-camera')
router.register(r'system-status', SystemStatusViewSet, basename='system-status')
router.register(r'subscription', SubscriptionViewSet, basename='subscription')
router.register(r'settings', SystemSettingViewSet, basename='settings')

urlpatterns = [
    path('', include(router.urls)),
]

# Available endpoints:
# GET    /admin/users/                        - List users (role-based)
# POST   /admin/users/                        - Create user  
# GET    /admin/users/{id}/                   - Retrieve user
# PUT    /admin/users/{id}/                   - Update user
# PATCH  /admin/users/{id}/                   - Partial update user
# DELETE /admin/users/{id}/                   - Delete user
# POST   /admin/users/add_role/               - Add new role (Admin only)
# GET    /admin/users/user_permissions/       - Get current user permissions
# POST   /admin/users/{id}/block/             - Block user
# POST   /admin/users/{id}/unblock/           - Unblock user
# POST   /admin/users/{id}/activate/          - Activate user
# POST   /admin/users/{id}/deactivate/        - Deactivate user
# POST   /admin/users/{id}/update_status/     - Update user status

# GET    /admin/cameras/                      - List all cameras (with pagination)
# GET    /admin/cameras/{id}/                 - Retrieve camera
# PUT    /admin/cameras/{id}/                 - Update camera
# DELETE /admin/cameras/{id}/                 - Delete camera

# GET    /admin/system-status/status/         - Get system status metrics
# GET    /admin/subscription/                 - List subscription plans
# GET    /admin/subscription/user_subscription/ - Get user subscription
# PUT    /admin/subscription/{id}/update_user_subscription/ - Update user subscription
# GET    /admin/settings/                     - List system settings
# GET    /admin/settings/by_category/         - Get settings by category