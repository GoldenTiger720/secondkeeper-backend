# Configuration for TensorRT detectors
# Based on YOLOv8-TensorRT-main config

# Class names for different models
FIRE_SMOKE_CLASSES = {
    0: 'fire',
    1: 'smoke'
}

# Default COCO classes (for compatibility)
COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
    27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
    32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
    45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
    50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
    55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
    60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
    65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
    69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
    74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',
    78: 'hair drier', 79: 'toothbrush'
}

# Use fire/smoke classes as default
CLASSES = FIRE_SMOKE_CLASSES

# Colors for visualization (BGR format)
FIRE_SMOKE_COLORS = {
    0: (0, 0, 255),    # Red for fire
    1: (128, 128, 128)  # Gray for smoke
}

# Default colors for visualization
COLORS = FIRE_SMOKE_COLORS

# Model configurations
MODEL_CONFIGS = {
    'fire_smoke': {
        'classes': FIRE_SMOKE_CLASSES,
        'colors': FIRE_SMOKE_COLORS,
        'input_size': (640, 640),
        'conf_threshold': 0.25,
        'iou_threshold': 0.45
    },
    'coco': {
        'classes': COCO_CLASSES,
        'colors': {i: (0, 255, 0) for i in range(80)},  # Green for all classes
        'input_size': (640, 640),
        'conf_threshold': 0.25,
        'iou_threshold': 0.45
    }
}

def get_model_config(model_name='fire_smoke'):
    """Get configuration for specific model"""
    return MODEL_CONFIGS.get(model_name, MODEL_CONFIGS['fire_smoke'])

def update_classes_and_colors(model_name='fire_smoke'):
    """Update global CLASSES and COLORS variables"""
    global CLASSES, COLORS
    config = get_model_config(model_name)
    CLASSES = config['classes']
    COLORS = config['colors']