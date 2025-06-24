import os
import sys
import numpy as np
import cv2
import torch
from PIL import Image
import logging
from pathlib import Path

# Add YOLOv8-TensorRT-main to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'YOLOv8-TensorRT-main'))

from models.engine import TRTModule
from models.torch_utils import det_postprocess
from models.utils import blob, letterbox
from .config import CLASSES, COLORS

logger = logging.getLogger('security_ai')

class TensorRTDetector:
    """TensorRT-based detector for optimized inference"""
    
    def __init__(self, engine_path, config_path=None, name="TensorRT Detector"):
        """Initialize TensorRT detector
        
        Args:
            engine_path: Path to TensorRT engine file
            config_path: Path to config JSON file (optional)
            name: Name of the detector
        """
        self.engine_path = engine_path
        self.config_path = config_path
        self.name = name
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_height = 640
        self.input_width = 640
        self.class_names = {}
        
        # Load configuration if available
        if config_path and os.path.exists(config_path):
            import json
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {}
    
    def load_model(self):
        """Load TensorRT engine"""
        if self.model is None:
            if not os.path.exists(self.engine_path):
                logger.error(f"TensorRT engine not found: {self.engine_path}")
                raise FileNotFoundError(f"TensorRT engine not found: {self.engine_path}")
            
            try:
                logger.info(f"Loading TensorRT engine: {self.engine_path}")
                
                # Use the working TRTModule from YOLOv8-TensorRT-main
                self.model = TRTModule(self.engine_path, self.device)
                
                # Get input dimensions from model
                if hasattr(self.model, 'inp_info') and len(self.model.inp_info) > 0:
                    input_shape = self.model.inp_info[0].shape
                    if len(input_shape) >= 2:
                        self.input_height, self.input_width = input_shape[-2:]
                
                # Set desired output order for detection models
                try:
                    self.model.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])
                except Exception as e:
                    logger.warning(f"Could not set desired output order: {e}")
                    # Continue anyway, the model might work with default output order
                
                logger.info(f"TensorRT model loaded successfully. Input size: {self.input_width}x{self.input_height}")
                
            except Exception as e:
                logger.error(f"Error loading TensorRT engine {self.engine_path}: {str(e)}")
                logger.error(f"Full error: {type(e).__name__}: {str(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise
        
        return self.model
    
    def preprocess_image(self, image):
        """Preprocess image for TensorRT inference
        
        Args:
            image: Input image (numpy array or PIL Image)
            
        Returns:
            Preprocessed tensor, ratio, padding info
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        elif isinstance(image, str):
            image = cv2.imread(image)
        
        # Letterbox resize
        processed_img, ratio, dwdh = letterbox(image, (self.input_width, self.input_height))
        
        # Convert BGR to RGB
        rgb_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor
        tensor = blob(rgb_img, return_seg=False)
        dwdh = torch.asarray(dwdh * 2, dtype=torch.float32, device=self.device)
        tensor = torch.asarray(tensor, device=self.device)
        
        return tensor, ratio, dwdh
    
    def postprocess_results(self, model_output, ratio, dwdh, conf_threshold=0.25):
        """Postprocess TensorRT model output
        
        Args:
            model_output: Raw output from TensorRT model
            ratio: Resize ratio from preprocessing
            dwdh: Padding info from preprocessing
            conf_threshold: Confidence threshold for filtering detections
            
        Returns:
            Processed bounding boxes, scores, labels
        """
        # Debug: check model output format (only log once)
        if not hasattr(self, '_output_format_logged'):
            logger.info(f"TensorRT model output format: {type(model_output)}")
            if isinstance(model_output, (tuple, list)):
                logger.info(f"Output count: {len(model_output)}")
                for i, output in enumerate(model_output):
                    logger.info(f"Output {i}: shape={output.shape if hasattr(output, 'shape') else 'N/A'}")
            else:
                logger.info(f"Output shape: {model_output.shape if hasattr(model_output, 'shape') else 'N/A'}")
            self._output_format_logged = True
        
        try:
            bboxes, scores, labels = det_postprocess(model_output)
        except (AssertionError, TypeError) as e:
            if not hasattr(self, '_postprocess_warning_logged'):
                logger.warning(f"Using custom postprocessing for TensorRT output format")
                self._postprocess_warning_logged = True
            # Custom postprocessing for single tensor output format [batch, num_dets, 6]
            if isinstance(model_output, torch.Tensor) and len(model_output.shape) == 3:
                # Output format: [batch, num_detections, 6] where 6 = [x1, y1, x2, y2, score, class]
                output = model_output[0]  # Remove batch dimension
                
                # Filter out invalid detections (score > conf_threshold)
                valid_mask = output[:, 4] > conf_threshold
                valid_output = output[valid_mask]
                
                if len(valid_output) == 0:
                    # No valid detections
                    return [], [], []
                
                # Extract components
                bboxes = valid_output[:, :4]  # [x1, y1, x2, y2]
                scores = valid_output[:, 4]   # confidence scores
                labels = valid_output[:, 5].long()  # class indices
                
                if len(bboxes) > 0:
                    logger.debug(f"TensorRT detected {len(bboxes)} objects")
                
            elif isinstance(model_output, (tuple, list)) and len(model_output) != 4:
                # Handle different output formats
                if len(model_output) == 1:
                    # Single output tensor in a tuple/list
                    logger.warning("Single output tensor in container detected")
                    return self.postprocess_results(model_output[0], ratio, dwdh)
                else:
                    logger.warning(f"Unexpected output format with {len(model_output)} tensors")
                    return [], [], []
            else:
                logger.error(f"Cannot handle output format: {type(model_output)}, shape: {model_output.shape if hasattr(model_output, 'shape') else 'N/A'}")
                return [], [], []
        
        if bboxes.numel() == 0:
            return [], [], []
        
        # Adjust coordinates back to original image space
        bboxes -= dwdh
        bboxes /= ratio
        
        return bboxes, scores, labels
    
    def predict_image(self, img, conf_threshold=0.25, iou_threshold=0.45, image_size=640):
        """Predict objects in an image using TensorRT
        
        Args:
            img: Input image
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold (not used in TensorRT postprocessing)
            image_size: Input image size (not used, model determines size)
            
        Returns:
            Annotated image, detection results
        """
        # Load model
        model = self.load_model()
        
        try:
            # Preprocess image
            tensor, ratio, dwdh = self.preprocess_image(img)
            
            # Run inference
            with torch.no_grad():
                model_output = model(tensor)
            
            # Postprocess results
            bboxes, scores, labels = self.postprocess_results(model_output, ratio, dwdh, conf_threshold)
            
            # Create annotated image
            if isinstance(img, str):
                draw_img = cv2.imread(img)
            elif isinstance(img, Image.Image):
                draw_img = np.array(img)
            else:
                draw_img = img.copy()
            
            # Draw detections
            annotated_img = self.draw_detections(draw_img, bboxes, scores, labels, conf_threshold)
            
            # Convert to PIL Image
            annotated_pil = Image.fromarray(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
            
            # Create results structure compatible with YOLO format
            results = self.create_yolo_compatible_results(bboxes, scores, labels, conf_threshold)
            
            return annotated_pil, results
            
        except Exception as e:
            logger.error(f"Error in TensorRT predict_image: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def predict_video_frame(self, detect_type, frame, conf_threshold=0.25, iou_threshold=0.45, image_size=640):
        """Process a single video frame using TensorRT
        
        Args:
            detect_type: Type of detection (for compatibility)
            frame: Input frame
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold
            image_size: Input image size
            
        Returns:
            Annotated frame, detection results
        """
        # Load model
        model = self.load_model()
        
        try:
            # Preprocess frame
            tensor, ratio, dwdh = self.preprocess_image(frame)
            
            # Run inference
            with torch.no_grad():
                model_output = model(tensor)
            
            # Postprocess results
            bboxes, scores, labels = self.postprocess_results(model_output, ratio, dwdh, conf_threshold)
            
            # Draw detections
            annotated_frame = self.draw_detections(frame.copy(), bboxes, scores, labels, conf_threshold)
            
            # Create results structure compatible with YOLO format
            results = self.create_yolo_compatible_results(bboxes, scores, labels, conf_threshold)
            
            return annotated_frame, results
            
        except Exception as e:
            logger.error(f"Error in TensorRT predict_video_frame: {str(e)}")
            raise
    
    def draw_detections(self, image, bboxes, scores, labels, conf_threshold):
        """Draw detection results on image
        
        Args:
            image: Input image
            bboxes: Bounding boxes
            scores: Confidence scores
            labels: Class labels
            conf_threshold: Confidence threshold
            
        Returns:
            Annotated image
        """
        if len(bboxes) == 0:
            return image
        
        for bbox, score, label in zip(bboxes, scores, labels):
            if score < conf_threshold:
                continue
                
            bbox = bbox.round().int().tolist()
            cls_id = int(label)
            
            # Get class name
            cls_name = CLASSES.get(cls_id, f"Class_{cls_id}")
            
            # Get color
            color = COLORS.get(cls_id, (0, 255, 0))
            
            # Draw bounding box
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Draw label
            label_text = f"{cls_name}: {score:.2f}"
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(image, (bbox[0], bbox[1] - label_size[1] - 10), 
                         (bbox[0] + label_size[0], bbox[1]), color, -1)
            cv2.putText(image, label_text, (bbox[0], bbox[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return image
    
    def create_yolo_compatible_results(self, bboxes, scores, labels, conf_threshold):
        """Create YOLO-compatible results structure
        
        Args:
            bboxes: Bounding boxes
            scores: Confidence scores
            labels: Class labels
            conf_threshold: Confidence threshold
            
        Returns:
            YOLO-compatible results
        """
        class YOLOBoxes:
            def __init__(self, bboxes, scores, labels, conf_threshold):
                self.data = []
                self.conf = []
                self.cls = []
                
                for bbox, score, label in zip(bboxes, scores, labels):
                    if score >= conf_threshold:
                        if hasattr(bbox, 'cpu'):
                            bbox_np = bbox.cpu().numpy()
                        else:
                            bbox_np = np.array(bbox) if not isinstance(bbox, np.ndarray) else bbox
                        
                        self.data.append(bbox_np)
                        self.conf.append(float(score))
                        self.cls.append(int(label))
                
                # Convert to numpy arrays or torch tensors (YOLO format)
                if self.data:
                    self.data = np.array(self.data)
                    if hasattr(bboxes, 'device'):  # If original was torch tensor
                        import torch
                        self.conf = torch.tensor(self.conf, device=bboxes.device)
                        self.cls = torch.tensor(self.cls, device=bboxes.device)
                    else:
                        self.conf = np.array(self.conf)
                        self.cls = np.array(self.cls)
                else:
                    self.data = np.array([]).reshape(0, 4)
                    self.conf = np.array([])
                    self.cls = np.array([])
            
            def __len__(self):
                return len(self.data)
                
            def tolist(self):
                return self.conf.tolist() if hasattr(self.conf, 'tolist') else list(self.conf)
        
        class YOLOResult:
            def __init__(self, bboxes, scores, labels, conf_threshold):
                self.boxes = YOLOBoxes(bboxes, scores, labels, conf_threshold)
        
        return [YOLOResult(bboxes, scores, labels, conf_threshold)]
    
    def get_description(self):
        """Return description of the detector"""
        return f"TensorRT optimized detector for high-performance inference. Engine: {os.path.basename(self.engine_path)}"
    
    def get_model_info(self):
        """Return model information"""
        return {
            "name": self.name,
            "engine_path": self.engine_path,
            "config_path": self.config_path,
            "input_size": f"{self.input_width}x{self.input_height}",
            "type": "TensorRT Engine",
            "device": str(self.device)
        }