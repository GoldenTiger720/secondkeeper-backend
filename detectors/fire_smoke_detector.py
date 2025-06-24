from .base_detector import BaseDetector
from .tensorrt_detector import TensorRTDetector
from django.conf import settings
import os
import logging

logger = logging.getLogger('security_ai')

class FireSmokeDetector(BaseDetector):
    """Detector specialized for fire and smoke detection with TensorRT optimization"""
    
    def __init__(self, use_tensorrt=True):
        """Initialize the fire and smoke detector
        
        Args:
            use_tensorrt: Whether to use TensorRT engine if available
        """
        self.use_tensorrt = use_tensorrt
        self.tensorrt_detector = None
        
        # Initialize base detector
        super().__init__(model_path=settings.MODEL_PATHS['fire_smoke'], name="Fire and Smoke")
        
        # Try to initialize TensorRT detector if requested and engine exists
        if use_tensorrt:
            self._try_init_tensorrt()
    
    def _try_init_tensorrt(self):
        """Try to initialize TensorRT detector"""
        try:
            # Construct paths for TensorRT engine
            models_dir = os.path.dirname(self.model_path)
            engine_path = os.path.join(models_dir, 'fire_smoke.engine')
            config_path = os.path.join(models_dir, 'fire_smoke_config.json')
            
            if os.path.exists(engine_path):
                logger.info(f"Initializing TensorRT detector with engine: {engine_path}")
                self.tensorrt_detector = TensorRTDetector(
                    engine_path=engine_path,
                    config_path=config_path if os.path.exists(config_path) else None,
                    name="Fire and Smoke TensorRT"
                )
                logger.info("TensorRT detector initialized successfully")
            else:
                logger.warning(f"TensorRT engine not found at {engine_path}, falling back to YOLO")
                self.use_tensorrt = False
                
        except Exception as e:
            logger.error(f"Failed to initialize TensorRT detector: {str(e)}")
            logger.info("Falling back to standard YOLO detector")
            self.use_tensorrt = False
            self.tensorrt_detector = None
    
    def predict_image(self, img, conf_threshold, iou_threshold, image_size):
        """Predict objects in an image using the best available detector"""
        if self.use_tensorrt and self.tensorrt_detector:
            try:
                return self.tensorrt_detector.predict_image(img, conf_threshold, iou_threshold, image_size)
            except Exception as e:
                logger.error(f"TensorRT prediction failed: {str(e)}")
                logger.info("Falling back to YOLO detector")
                # Fall back to base detector
                return super().predict_image(img, conf_threshold, iou_threshold, image_size)
        else:
            return super().predict_image(img, conf_threshold, iou_threshold, image_size)
    
    def predict_video_frame(self, detect_type, frame, conf_threshold, iou_threshold, image_size):
        """Process a single video frame using the best available detector"""
        if self.use_tensorrt and self.tensorrt_detector:
            try:
                return self.tensorrt_detector.predict_video_frame(detect_type, frame, conf_threshold, iou_threshold, image_size)
            except Exception as e:
                logger.error(f"TensorRT frame prediction failed: {str(e)}")
                logger.info("Falling back to YOLO detector")
                # Fall back to base detector
                return super().predict_video_frame(detect_type, frame, conf_threshold, iou_threshold, image_size)
        else:
            return super().predict_video_frame(detect_type, frame, conf_threshold, iou_threshold, image_size)
    
    def get_description(self):
        """Return a description of the detector"""
        base_desc = "Detects fire and smoke in images and videos. Can help in early detection of fire incidents."
        if self.use_tensorrt and self.tensorrt_detector:
            return base_desc + " Uses TensorRT optimization for faster inference."
        else:
            return base_desc + " Uses standard YOLO implementation."
    
    def get_model_info(self):
        """Return information about the model used by the detector"""
        base_info = {
            "name": self.name,
            "path": self.model_path,
            "classes": self.class_names,
        }
        
        if self.use_tensorrt and self.tensorrt_detector:
            tensorrt_info = self.tensorrt_detector.get_model_info()
            base_info.update({
                "type": "TensorRT Optimized YOLOv8",
                "engine_path": tensorrt_info.get("engine_path", ""),
                "input_size": tensorrt_info.get("input_size", ""),
                "device": tensorrt_info.get("device", "")
            })
        else:
            base_info["type"] = "YOLOv8 Object Detection"
        
        return base_info
    
    def is_using_tensorrt(self):
        """Check if currently using TensorRT"""
        return self.use_tensorrt and self.tensorrt_detector is not None