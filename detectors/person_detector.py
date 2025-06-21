from .base_detector import BaseDetector
from django.conf import settings

class PersonDetector(BaseDetector):
    """Detector specialized for person detection using YOLOv8m"""
    
    def __init__(self):
        """Initialize the person detector"""
        super().__init__(model_path=settings.MODEL_PATHS['person'], name="Person Detection")
    
    def get_description(self):
        """Return a description of the detector"""
        return "Detects people in images and videos using YOLOv8m model."
    
    def get_model_info(self):
        """Return information about the model used by the detector"""
        return {
            "name": self.name,
            "path": self.model_path,
            "classes": self.class_names,
            "type": "YOLOv8m Object Detection"
        }