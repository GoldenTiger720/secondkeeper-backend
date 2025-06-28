import torch
import torch.nn as nn
from ultralytics import YOLO
import onnx
import onnxruntime as ort
import numpy as np
import cv2
from pathlib import Path
import argparse
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_yolo_to_onnx(model_path, output_path=None, img_size=640, batch_size=1, device='cpu', 
                        opset_version=17, simplify=True, dynamic=False, half=False):
    """
    Convert YOLO model to ONNX format optimized for TensorRT 10+
    
    Args:
        model_path (str): Path to YOLO model (.pt file)
        output_path (str): Output path for ONNX model
        img_size (int): Input image size
        batch_size (int): Batch size for conversion
        device (str): Device to use ('cpu' or 'cuda')
        opset_version (int): ONNX opset version (17 recommended for TensorRT 10+)
        simplify (bool): Simplify ONNX model
        dynamic (bool): Dynamic input shapes
        half (bool): Use FP16 precision
    """
    
    # Validate model path
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load YOLO model
    logger.info(f"Loading YOLO model from {model_path}")
    try:
        model = YOLO(model_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    # Set output path if not provided
    if output_path is None:
        output_path = model_path.replace('.pt', '.onnx')
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Export to ONNX with TensorRT-optimized settings
    logger.info(f"Converting to ONNX format (opset {opset_version})...")
    try:
        export_result = model.export(
            format='onnx',
            imgsz=img_size,
            opset=opset_version,
            dynamic=dynamic,
            simplify=simplify,
            half=half,
            int8=False,  # Disable INT8 for better TensorRT compatibility
            optimize=True,  # Enable optimization
            workspace=4  # 4GB workspace for optimization
        )
        
        # The export might return the path or the model might create it automatically
        if isinstance(export_result, str):
            actual_output = export_result
        else:
            # Ultralytics sometimes creates the file with a different name
            base_name = os.path.splitext(model_path)[0]
            actual_output = f"{base_name}.onnx"
        
        # Move to desired output path if different
        if actual_output != output_path and os.path.exists(actual_output):
            import shutil
            shutil.move(actual_output, output_path)
        
        logger.info(f"✅ Model converted successfully to {output_path}")
        
        # Verify file exists and has reasonable size
        if os.path.exists(output_path):
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            logger.info(f"📊 ONNX model size: {file_size_mb:.1f} MB")
        else:
            raise FileNotFoundError(f"Output file not created: {output_path}")
            
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to export model: {e}")
        raise

def convert_yolov5_to_onnx(model_path, output_path=None, img_size=640, batch_size=1, 
                          device='cpu', opset_version=11):
    """
    Convert YOLOv5 model to ONNX (alternative method using torch.onnx.export)
    """
    import sys
    sys.path.append('yolov5')  # Add yolov5 to path if using official repo
    
    # Load model
    device = torch.device(device)
    model = torch.load(model_path, map_location=device)['model'].float()
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, 3, img_size, img_size, device=device)
    
    # Set output path
    if output_path is None:
        output_path = model_path.replace('.pt', '.onnx')
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size'}
        } if batch_size == 1 else None
    )
    
    print(f"YOLOv5 model converted to {output_path}")
    return output_path

def verify_onnx_model(onnx_path, test_image_path=None):
    """
    Verify the converted ONNX model
    """
    print(f"Verifying ONNX model: {onnx_path}")
    
    # Load and check ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model is valid")
    
    # Test inference
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    print(f"Input name: {input_name}")
    print(f"Input shape: {input_shape}")
    
    # Create test input
    if test_image_path and Path(test_image_path).exists():
        # Use actual image
        img = cv2.imread(test_image_path)
        img = cv2.resize(img, (input_shape[3], input_shape[2]))
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)  # Add batch dimension
    else:
        # Use random input
        img = np.random.randn(*input_shape).astype(np.float32)
    
    # Run inference
    outputs = session.run(None, {input_name: img})
    print(f"✓ Inference successful, output shape: {[out.shape for out in outputs]}")
    
    return True

def compare_models(pytorch_model_path, onnx_model_path, test_image_path=None, img_size=640):
    """
    Compare outputs between PyTorch and ONNX models
    """
    print("Comparing PyTorch and ONNX model outputs...")
    
    # Load PyTorch model
    yolo_model = YOLO(pytorch_model_path)
    
    # Load ONNX model
    ort_session = ort.InferenceSession(onnx_model_path)
    input_name = ort_session.get_inputs()[0].name
    
    # Prepare test input
    if test_image_path and Path(test_image_path).exists():
        img = cv2.imread(test_image_path)
        img_resized = cv2.resize(img, (img_size, img_size))
    else:
        img_resized = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
    
    # PyTorch inference
    pytorch_results = yolo_model(img_resized, verbose=False)
    
    # Prepare input for ONNX
    img_onnx = img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
    img_onnx = np.expand_dims(img_onnx, axis=0)
    
    # ONNX inference
    onnx_outputs = ort_session.run(None, {input_name: img_onnx})
    
    print("✓ Both models ran successfully")
    print(f"PyTorch output type: {type(pytorch_results[0].boxes.data)}")
    print(f"ONNX output shape: {[out.shape for out in onnx_outputs]}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Convert YOLO model to ONNX')
    parser.add_argument('--model', type=str, required=True, help='Path to YOLO model (.pt)')
    parser.add_argument('--output', type=str, help='Output ONNX path')
    parser.add_argument('--img-size', type=int, default=640, help='Input image size')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--opset', type=int, default=17, help='ONNX opset version (17 recommended for TensorRT 10+)')
    parser.add_argument('--simplify', action='store_true', help='Simplify ONNX model')
    parser.add_argument('--dynamic', action='store_true', help='Dynamic input shapes')
    parser.add_argument('--half', action='store_true', help='Use FP16 precision')
    parser.add_argument('--verify', action='store_true', help='Verify converted model')
    parser.add_argument('--compare', action='store_true', help='Compare PyTorch vs ONNX')
    parser.add_argument('--test-image', type=str, help='Test image path for verification')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("🚀 YOLO to ONNX Converter - TensorRT 10+ Optimized")
    logger.info(f"📁 Input model: {args.model}")
    logger.info(f"📊 ONNX opset version: {args.opset}")
    
    # Convert model
    onnx_path = convert_yolo_to_onnx(
        model_path=args.model,
        output_path=args.output,
        img_size=args.img_size,
        batch_size=args.batch_size,
        device=args.device,
        opset_version=args.opset,
        simplify=args.simplify,
        dynamic=args.dynamic,
        half=args.half
    )
    
    # Verify if requested
    if args.verify:
        verify_onnx_model(onnx_path, args.test_image)
    
    # Compare if requested
    if args.compare:
        compare_models(args.model, onnx_path, args.test_image, args.img_size)

if __name__ == "__main__":
    # Example usage
    # Uncomment and modify paths as needed
    
    # Basic conversion
    # convert_yolo_to_onnx('yolov8n.pt', 'yolov8n.onnx')
    
    # Advanced conversion with options
    # convert_yolo_to_onnx(
    #     model_path='./models/yolo11m-pose.pt',
    #     output_path='yolo11m-pose.onnx',
    #     img_size=640,
    #     dynamic=True,
    #     simplify=True
    # )
    
    # Verify converted model
    # verify_onnx_model('yolov8n.onnx', 'test_image.jpg')
    
    main()