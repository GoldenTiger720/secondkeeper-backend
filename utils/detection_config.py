# utils/detection_config.py

import json
import logging
from django.core.cache import cache
from admin_panel.models import SystemSetting
from django.conf import settings

logger = logging.getLogger('security_ai')

class DetectionConfigManager:
    """
    Manages configuration for the detection service
    """
    
    CACHE_KEY_PREFIX = 'detection_config:'
    CACHE_TIMEOUT = 300  # 5 minutes
    
    DEFAULT_CONFIG = {
        # Performance settings
        'frame_skip': 5,  # Process every Nth frame
        'max_concurrent_cameras': 20,
        'detection_cooldown': 30,  # Seconds between alerts for same camera
        'video_clip_duration': 10,  # Seconds
        
        # Detection thresholds
        'confidence_threshold': 0.6,
        'iou_threshold': 0.45,
        'image_size': 640,
        
        # GPU settings
        'use_gpu': True,
        'gpu_batch_size': 4,
        'gpu_memory_fraction': 0.8,
        
        # Alert settings
        'auto_create_alerts': True,
        'auto_send_notifications': True,
        'save_detection_videos': True,
        'save_detection_thumbnails': True,
        
        # Health monitoring
        'health_check_interval': 60,
        'cpu_warning_threshold': 80,
        'memory_warning_threshold': 80,
        'disk_warning_threshold': 85,
        
        # Logging
        'log_level': 'INFO',
        'log_detection_stats': True,
        'log_performance_metrics': True,
        
        # Cleanup settings
        'cleanup_old_videos_days': 30,
        'cleanup_old_thumbnails_days': 60,
        'cleanup_old_logs_days': 7,
        
        # Advanced settings
        'enable_preprocessing': False,
        'preprocessing_resize': True,
        'preprocessing_normalize': True,
        'enable_postprocessing': True,
        'postprocessing_nms': True,
        'postprocessing_filter_small': True
    }
    
    def __init__(self):
        self.config = self.DEFAULT_CONFIG.copy()
        self.load_config()
        
    def load_config(self):
        """Load configuration from database and cache"""
        try:
            # Try to load from cache first
            cached_config = cache.get(f"{self.CACHE_KEY_PREFIX}main")
            if cached_config:
                self.config.update(cached_config)
                return
                
            # Load from database
            db_config = {}
            settings_qs = SystemSetting.objects.filter(
                category='detection',
                is_editable=True
            )
            
            for setting in settings_qs:
                try:
                    value = self._parse_setting_value(setting)
                    db_config[setting.key] = value
                except Exception as e:
                    logger.error(f"Error parsing setting {setting.key}: {str(e)}")
                    
            # Update config with database values
            self.config.update(db_config)
            
            # Cache the config
            cache.set(f"{self.CACHE_KEY_PREFIX}main", db_config, self.CACHE_TIMEOUT)
            
            logger.info("Detection configuration loaded from database")
            
        except Exception as e:
            logger.error(f"Error loading detection config: {str(e)}")
            logger.info("Using default configuration")
            
    def _parse_setting_value(self, setting):
        """Parse setting value based on data type"""
        if setting.data_type == 'integer':
            return int(setting.value)
        elif setting.data_type == 'float':
            return float(setting.value)
        elif setting.data_type == 'boolean':
            return setting.value.lower() in ('true', 'yes', '1', 't', 'y')
        elif setting.data_type == 'json':
            return json.loads(setting.value)
        else:
            return setting.value
            
    def get(self, key, default=None):
        """Get configuration value"""
        return self.config.get(key, default)
        
    def set(self, key, value, save_to_db=True):
        """Set configuration value"""
        self.config[key] = value
        
        if save_to_db:
            try:
                self._save_setting_to_db(key, value)
                # Invalidate cache
                cache.delete(f"{self.CACHE_KEY_PREFIX}main")
            except Exception as e:
                logger.error(f"Error saving setting {key} to database: {str(e)}")
                
    def _save_setting_to_db(self, key, value):
        """Save setting to database"""
        # Determine data type
        data_type = 'string'
        if isinstance(value, bool):
            data_type = 'boolean'
            value = str(value).lower()
        elif isinstance(value, int):
            data_type = 'integer'
            value = str(value)
        elif isinstance(value, float):
            data_type = 'float'
            value = str(value)
        elif isinstance(value, (dict, list)):
            data_type = 'json'
            value = json.dumps(value)
        else:
            value = str(value)
            
        # Create or update setting
        SystemSetting.objects.update_or_create(
            key=key,
            defaults={
                'value': value,
                'data_type': data_type,
                'category': 'detection',
                'is_editable': True,
                'description': f'Detection service setting: {key}'
            }
        )
        
    def reload_config(self):
        """Reload configuration from database"""
        cache.delete(f"{self.CACHE_KEY_PREFIX}main")
        self.load_config()
        
    def get_all_settings(self):
        """Get all configuration settings"""
        return self.config.copy()
        
    def reset_to_defaults(self):
        """Reset configuration to defaults"""
        self.config = self.DEFAULT_CONFIG.copy()
        
        # Delete all detection settings from database
        SystemSetting.objects.filter(category='detection').delete()
        
        # Clear cache
        cache.delete(f"{self.CACHE_KEY_PREFIX}main")
        
    def validate_config(self):
        """Validate configuration values"""
        errors = []
        
        # Validate numeric ranges
        if not 1 <= self.get('frame_skip', 1) <= 30:
            errors.append("frame_skip must be between 1 and 30")
            
        if not 0.1 <= self.get('confidence_threshold', 0.6) <= 1.0:
            errors.append("confidence_threshold must be between 0.1 and 1.0")
            
        if not 0.1 <= self.get('iou_threshold', 0.45) <= 1.0:
            errors.append("iou_threshold must be between 0.1 and 1.0")
            
        if not 320 <= self.get('image_size', 640) <= 1280:
            errors.append("image_size must be between 320 and 1280")
            
        if not 1 <= self.get('max_concurrent_cameras', 20) <= 100:
            errors.append("max_concurrent_cameras must be between 1 and 100")
            
        if not 5 <= self.get('detection_cooldown', 30) <= 3600:
            errors.append("detection_cooldown must be between 5 and 3600 seconds")
            
        if not 5 <= self.get('video_clip_duration', 10) <= 300:
            errors.append("video_clip_duration must be between 5 and 300 seconds")
            
        # Validate GPU settings
        if self.get('use_gpu', True) and not torch.cuda.is_available():
            errors.append("GPU not available but use_gpu is enabled")
            
        return errors
        
    def apply_runtime_optimizations(self):
        """Apply runtime optimizations based on system capabilities"""
        try:
            import psutil
            import torch
            
            # Adjust settings based on available resources
            cpu_count = psutil.cpu_count()
            memory_gb = psutil.virtual_memory().total / (1024**3)
            
            # Adjust frame skip based on CPU cores
            if cpu_count <= 2:
                self.set('frame_skip', 10, save_to_db=False)
            elif cpu_count <= 4:
                self.set('frame_skip', 7, save_to_db=False)
            else:
                self.set('frame_skip', 5, save_to_db=False)
                
            # Adjust max cameras based on memory
            if memory_gb < 4:
                self.set('max_concurrent_cameras', 5, save_to_db=False)
            elif memory_gb < 8:
                self.set('max_concurrent_cameras', 10, save_to_db=False)
            elif memory_gb < 16:
                self.set('max_concurrent_cameras', 15, save_to_db=False)
            else:
                self.set('max_concurrent_cameras', 20, save_to_db=False)
                
            # Adjust GPU settings
            if torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                if gpu_memory_gb < 4:
                    self.set('gpu_batch_size', 2, save_to_db=False)
                    self.set('image_size', 512, save_to_db=False)
                elif gpu_memory_gb < 8:
                    self.set('gpu_batch_size', 4, save_to_db=False)
                    self.set('image_size', 640, save_to_db=False)
                else:
                    self.set('gpu_batch_size', 8, save_to_db=False)
                    self.set('image_size', 640, save_to_db=False)
            else:
                self.set('use_gpu', False, save_to_db=False)
                self.set('image_size', 512, save_to_db=False)
                
            logger.info("Runtime optimizations applied to detection configuration")
            
        except Exception as e:
            logger.error(f"Error applying runtime optimizations: {str(e)}")
            
    def export_config(self):
        """Export configuration to JSON"""
        return json.dumps(self.config, indent=2)
        
    def import_config(self, config_json):
        """Import configuration from JSON"""
        try:
            imported_config = json.loads(config_json)
            
            # Validate imported config
            for key, value in imported_config.items():
                if key in self.DEFAULT_CONFIG:
                    self.set(key, value)
                else:
                    logger.warning(f"Unknown configuration key ignored: {key}")
                    
            logger.info("Configuration imported successfully")
            
        except Exception as e:
            logger.error(f"Error importing configuration: {str(e)}")
            raise


# Global configuration instance
detection_config = DetectionConfigManager()