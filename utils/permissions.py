# utils/permissions.py - Enhanced permissions with comprehensive role-based access control

from rest_framework import permissions

class IsOwnerOrAdmin(permissions.BasePermission):
    """
    Custom permission to only allow owners of an object or admins to access it.
    """
    
    def has_permission(self, request, view):
        """Check basic authentication and active status."""
        return bool(
            request.user and 
            request.user.is_authenticated and 
            request.user.is_active
        )
    
    def has_object_permission(self, request, view, obj):
        """Check object-level permissions."""
        # Check if the user is an admin - admins have access to everything
        if request.user.is_admin():
            return True
        
        # Check if the object has a user attribute and if it equals the request user
        if hasattr(obj, 'user'):
            return obj.user == request.user
        
        # If the object is a user, check if it is the request user
        if hasattr(obj, 'email') and hasattr(obj, 'id'):
            return obj == request.user
        
        return False

class IsAdminUser(permissions.BasePermission):
    """
    Custom permission to only allow admin users to access the view.
    """
    
    def has_permission(self, request, view):
        return bool(
            request.user and 
            request.user.is_authenticated and 
            request.user.is_active and
            request.user.is_admin()
        )

class IsManagerOrAdminOrReviewer(permissions.BasePermission):
    """
    Custom permission to only allow managers or admins to access the view.
    """
    
    def has_permission(self, request, view):
        return bool(
            request.user and 
            request.user.is_authenticated and 
            request.user.is_active and
            request.user.role in ['admin', 'manager', 'reviewer']
        )

class CanAddRoles(permissions.BasePermission):
    """
    Custom permission to only allow admin users to add new roles.
    """
    
    def has_permission(self, request, view):
        return bool(
            request.user and 
            request.user.is_authenticated and 
            request.user.is_active and
            request.user.can_add_roles()
        )

class IsReviewerOrAbove(permissions.BasePermission):
    """
    Custom permission for reviewers, managers, and admins.
    """
    
    def has_permission(self, request, view):
        return bool(
            request.user and 
            request.user.is_authenticated and 
            request.user.is_active and
            request.user.role in ['admin', 'manager', 'reviewer']
        )

class CanManageUsers(permissions.BasePermission):
    """
    Custom permission for users who can manage other users (admin and manager).
    """
    
    def has_permission(self, request, view):
        return bool(
            request.user and 
            request.user.is_authenticated and 
            request.user.is_active and
            request.user.can_manage_users()
        )

class IsOwnerOrCanManage(permissions.BasePermission):
    """
    Custom permission for object access - owner can access their own objects,
    managers can access user objects, admins can access all objects.
    """
    
    def has_permission(self, request, view):
        """Check basic authentication and active status."""
        return bool(
            request.user and 
            request.user.is_authenticated and 
            request.user.is_active
        )
    
    def has_object_permission(self, request, view, obj):
        """Check object-level permissions based on role hierarchy."""
        user = request.user
        
        # Admins can access everything
        if user.is_admin():
            return True
        
        # Check if object belongs to the user
        if hasattr(obj, 'user') and obj.user == user:
            return True
        
        # If the object is a user object
        if hasattr(obj, 'email') and hasattr(obj, 'role'):
            # Users can only access their own profile
            if obj == user:
                return True
            
            # Managers can manage regular users only
            if user.is_manager() and obj.role == 'user':
                return True
        
        # Check for camera objects - managers can manage cameras of regular users
        if hasattr(obj, 'user') and user.is_manager():
            if hasattr(obj.user, 'role') and obj.user.role == 'user':
                return True
        
        return False

class ReadOnlyOrOwnerWrite(permissions.BasePermission):
    """
    Custom permission to allow read-only access to everyone,
    but write access only to owners or admins.
    """
    
    def has_permission(self, request, view):
        """Check basic authentication for any access."""
        return bool(
            request.user and 
            request.user.is_authenticated and 
            request.user.is_active
        )
    
    def has_object_permission(self, request, view, obj):
        """Allow read access to all, write access to owners/admins only."""
        # Read permissions for everyone
        if request.method in permissions.SAFE_METHODS:
            return True
        
        # Write permissions only for owners or admins
        if request.user.is_admin():
            return True
        
        if hasattr(obj, 'user'):
            return obj.user == request.user
        
        if hasattr(obj, 'email'):
            return obj == request.user
        
        return False

class IsSelfOrAdmin(permissions.BasePermission):
    """
    Custom permission to allow users to access/modify their own data,
    or allow admins to access/modify any user data.
    """
    
    def has_permission(self, request, view):
        return bool(
            request.user and 
            request.user.is_authenticated and 
            request.user.is_active
        )
    
    def has_object_permission(self, request, view, obj):
        # Admins can access any user data
        if request.user.is_admin():
            return True
        
        # Users can only access their own data
        return obj == request.user

class IsActiveUser(permissions.BasePermission):
    """
    Custom permission to only allow active users.
    """
    
    def has_permission(self, request, view):
        return bool(
            request.user and 
            request.user.is_authenticated and 
            request.user.is_active and
            getattr(request.user, 'status', 'active') == 'active'
        )

class CanAccessAlerts(permissions.BasePermission):
    """
    Custom permission for alert access - all authenticated roles can view alerts.
    """
    
    def has_permission(self, request, view):
        return bool(
            request.user and 
            request.user.is_authenticated and 
            request.user.is_active and
            request.user.role in ['admin', 'manager', 'reviewer', 'user']
        )
    
    def has_object_permission(self, request, view, obj):
        user = request.user
        
        # Admins and reviewers can access all alerts
        if user.role in ['admin', 'reviewer']:
            return True
        
        # Managers can access alerts from users they manage
        if user.is_manager():
            # If alert belongs to a camera, check the camera's user
            if hasattr(obj, 'camera') and hasattr(obj.camera, 'user'):
                return obj.camera.user.role == 'user'
        
        # Users can only access their own alerts
        if hasattr(obj, 'camera') and hasattr(obj.camera, 'user'):
            return obj.camera.user == user
        
        return False

class CanModifySystemSettings(permissions.BasePermission):
    """
    Custom permission for system settings - only admins can modify.
    """
    
    def has_permission(self, request, view):
        # Only admins can modify system settings
        if request.method in permissions.SAFE_METHODS:
            # Read access for managers and admins
            return bool(
                request.user and 
                request.user.is_authenticated and 
                request.user.is_active and
                request.user.role in ['admin', 'manager']
            )
        else:
            # Write access only for admins
            return bool(
                request.user and 
                request.user.is_authenticated and 
                request.user.is_active and
                request.user.is_admin()
            )