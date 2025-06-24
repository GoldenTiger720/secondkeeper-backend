import os
import glob
import json
import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import onnx
from pathlib import Path
import argparse

class TensorRTConverter:
    def __init__(self, logger_level=trt.Logger.INFO):
        """Initialize TensorRT converter with logger"""
        self.logger = trt.Logger(logger_level)
        self.builder = trt.Builder(self.logger)
        self.config = self.builder.create_builder_config()
        self.network = None
        self.parser = None
        
    def preprocess_onnx_model(self, onnx_path):
        """Preprocess ONNX model to fix common TensorRT conversion issues"""
        print("🔧 Preprocessing ONNX model for TensorRT compatibility...")
        
        try:
            import onnx
            from onnx import helper, numpy_helper
            import onnx.optimizer
            
            # Load model
            model = onnx.load(onnx_path)
            
            # Apply ONNX optimizations that help with TensorRT
            passes = [
                'eliminate_deadend',
                'eliminate_duplicate_initializer',
                'eliminate_identity',
                'eliminate_nop_dropout',
                'eliminate_nop_monotone_argmax',
                'eliminate_nop_pad',
                'eliminate_nop_transpose',
                'eliminate_unused_initializer',
                'extract_constant_to_initializer',
                'fuse_add_bias_into_conv',
                'fuse_bn_into_conv',
                'fuse_consecutive_concats',
                'fuse_consecutive_log_softmax',
                'fuse_consecutive_reduce_unsqueeze',
                'fuse_consecutive_squeezes',
                'fuse_consecutive_transposes',
                'fuse_matmul_add_bias_into_gemm',
                'fuse_pad_into_conv',
                'fuse_transpose_into_gemm',
                'lift_lexical_references'
            ]
            
            # Apply optimizations
            optimized_model = onnx.optimizer.optimize(model, passes)
            
            # Save optimized model
            optimized_path = onnx_path.replace('.onnx', '_optimized.onnx')
            onnx.save(optimized_model, optimized_path)
            
            print(f"✅ Optimized ONNX model saved to: {optimized_path}")
            return optimized_path
            
        except Exception as e:
            print(f"⚠️  ONNX preprocessing failed: {e}")
            print("📝 Continuing with original model...")
            return onnx_path
    def get_optimal_config(self, onnx_path, gpu_memory_gb=None):
        """Determine optimal TensorRT configuration based on model and hardware"""
        
        # Get model info
        onnx_model = onnx.load(onnx_path)
        model_name = Path(onnx_path).stem.lower()
        file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        
        # Get GPU info
        device = cuda.Device(0)
        gpu_name = device.name()
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode()
        total_memory = device.total_memory() / (1024**3)  # GB
        
        # Default optimal configuration with conservative settings for problematic models
        config = {
            'max_workspace_size': min(2, int(total_memory * 0.3)) * (1024**3),  # Reduced workspace
            'fp16': True,  # Start with FP32 for stability
            'int8': True,  # Disable INT8 initially
            'tf32': True,  # Disable TF32 for problematic models
            'dla_core': None,
            'strict_types': True,  # Enable strict types for better compatibility
            'max_batch_size': 1,
            'optimization_level': 3,  # Reduced optimization level
            'quantization_flags': [],
            'calibration_dataset_size': 100,  # Reduced for faster processing
            'enable_plugins': True,
            'verbose': True
        }
        
        # Adjust based on GPU architecture
        compute_capability = device.compute_capability()
        
        # Only enable advanced features if model seems compatible
        if 'optimized' in onnx_path or 'yolov8' not in model_name:
            if compute_capability >= (7, 5):  # RTX 20xx series and newer
                config['fp16'] = True
                config['tf32'] = True
            elif compute_capability >= (7, 0):  # GTX 10xx series
                config['fp16'] = True
        
        # Conservative settings for YOLO models with known issues
        if 'yolo' in model_name:
            config['optimization_level'] = 2  # Lower optimization
            config['strict_types'] = True
            config['max_workspace_size'] = 1 * (1024**3)  # 1GB only
            
            if 'v8' in model_name or 'v11' in model_name:
                # YOLOv8/v11 often have activation fusion issues
                config['fp16'] = False  # Start with FP32
                config['tf32'] = False
        
        print(f"🎯 Conservative configuration for {gpu_name}:")
        print(f"   Compute Capability: {compute_capability}")
        print(f"   GPU Memory: {total_memory:.1f} GB")
        print(f"   Workspace Size: {config['max_workspace_size'] / (1024**3):.1f} GB")
        print(f"   FP16: {config['fp16']}")
        print(f"   INT8: {config['int8']}")
        print(f"   TF32: {config['tf32']}")
        print(f"   Optimization Level: {config['optimization_level']}")
        print(f"   Strict Types: {config['strict_types']}")
        
        return config
    
    def create_calibration_dataset(self, input_shape, dataset_size=500):
        """Create calibration dataset for INT8 quantization"""
        print(f"📊 Creating calibration dataset with {dataset_size} samples...")
        
        # Generate realistic calibration data
        calibration_data = []
        
        for i in range(dataset_size):
            # Create realistic image data (normalized to 0-1 range)
            if len(input_shape) == 4:  # NCHW format
                sample = np.random.rand(*input_shape).astype(np.float32)
            else:
                sample = np.random.rand(*input_shape).astype(np.float32)
            
            calibration_data.append(sample)
        
        return calibration_data
    
    def create_int8_calibrator(self, calibration_data, cache_file="calibration.cache"):
        """Create INT8 calibrator for quantization"""
        
        class PythonEntropyCalibrator(trt.IInt8EntropyCalibrator2):
            def __init__(self, data, cache_file):
                trt.IInt8EntropyCalibrator2.__init__(self)
                self.data = data
                self.cache_file = cache_file
                self.current_index = 0
                self.device_input = None
                
            def get_batch_size(self):
                return 1
                
            def get_batch(self, names):
                if self.current_index >= len(self.data):
                    return None
                    
                # Allocate device memory if not done
                if self.device_input is None:
                    self.device_input = cuda.mem_alloc(self.data[0].nbytes)
                
                # Copy data to device
                batch = self.data[self.current_index]
                cuda.memcpy_htod(self.device_input, batch)
                self.current_index += 1
                
                return [self.device_input]
                
            def read_calibration_cache(self):
                if os.path.exists(self.cache_file):
                    with open(self.cache_file, "rb") as f:
                        return f.read()
                return None
                
            def write_calibration_cache(self, cache):
                with open(self.cache_file, "wb") as f:
                    f.write(cache)
        
        return PythonEntropyCalibrator(calibration_data, cache_file)
    
    def build_engine(self, onnx_path, engine_path=None, config_override=None):
        """Build TensorRT engine from ONNX model with multiple fallback strategies"""
        
        if engine_path is None:
            engine_path = onnx_path.replace('.onnx', '.engine')
        
        print(f"🚀 Building TensorRT engine from {onnx_path}")
        
        # Try preprocessing first
        processed_onnx = self.preprocess_onnx_model(onnx_path)
        
        # Get optimal configuration
        config = self.get_optimal_config(processed_onnx)
        if config_override:
            config.update(config_override)
        
        # Try multiple build strategies in order of preference
        strategies = [
            ("Conservative FP32", {"fp16": True, "int8": True, "tf32": True, "optimization_level": 2}),
            ("FP16 only", {"fp16": True, "int8": True, "tf32": True, "optimization_level": 3}),
            ("FP16 + TF32", {"fp16": True, "int8": True, "tf32": True, "optimization_level": 3}),
            ("Original config", {})
        ]
        
        for strategy_name, strategy_config in strategies:
            print(f"\n🔄 Trying strategy: {strategy_name}")
            
            # Apply strategy config
            current_config = config.copy()
            current_config.update(strategy_config)
            
            try:
                result = self._build_engine_with_config(processed_onnx, engine_path, current_config)
                if result:
                    print(f"✅ Success with strategy: {strategy_name}")
                    return result
            except Exception as e:
                print(f"❌ Strategy '{strategy_name}' failed: {e}")
                continue
        
        print("❌ All build strategies failed")
        return None
    
    def _build_engine_with_config(self, onnx_path, engine_path, config):
        """Internal method to build engine with specific configuration"""
        
        # Create network
        explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = self.builder.create_network(explicit_batch)
        parser = trt.OnnxParser(network, self.logger)
        
        # Parse ONNX model
        print("📖 Parsing ONNX model...")
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                print("❌ Failed to parse ONNX model")
                for error in range(parser.num_errors):
                    print(f"   Error {error}: {parser.get_error(error)}")
                return None
        
        # Configure builder
        builder_config = self.builder.create_builder_config()
        
        # Set workspace/memory limit (compatibility with different TensorRT versions)
        if hasattr(builder_config, 'max_workspace_size'):
            # TensorRT < 8.5
            builder_config.max_workspace_size = config['max_workspace_size']
        else:
            # TensorRT >= 8.5
            builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, config['max_workspace_size'])
        
        # Set optimization level (compatibility check)
        if hasattr(builder_config, 'set_builder_optimization_level'):
            builder_config.set_builder_optimization_level(config['optimization_level'])
        elif hasattr(builder_config, 'builder_optimization_level'):
            builder_config.builder_optimization_level = config['optimization_level']
        
        # Enable strict types for better compatibility
        if config.get('strict_types', False):
            if hasattr(trt.BuilderFlag, 'STRICT_TYPES'):
                builder_config.set_flag(trt.BuilderFlag.STRICT_TYPES)
        
        # Enable precision modes with compatibility checks
        if config['fp16']:
            print("⚡ Enabling FP16 precision")
            if hasattr(trt.BuilderFlag, 'FP16'):
                builder_config.set_flag(trt.BuilderFlag.FP16)
            else:
                builder_config.flags |= 1 << int(trt.BuilderFlag.FP16)
        
        if config['tf32']:
            print("⚡ Enabling TF32 precision")
            if hasattr(trt.BuilderFlag, 'TF32'):
                builder_config.set_flag(trt.BuilderFlag.TF32)
        
        # Skip INT8 for now due to calibration complexity
        # Will add back after basic conversion works
        
        # Configure DLA if available (compatibility check)
        if config['dla_core'] is not None:
            print(f"🔧 Enabling DLA core {config['dla_core']}")
            if hasattr(builder_config, 'default_device_type'):
                builder_config.default_device_type = trt.DeviceType.DLA
                builder_config.DLA_core = config['dla_core']
            else:
                print("⚠️  DLA not supported in this TensorRT version")
        
        # Set dynamic shapes if needed
        input_tensor = network.get_input(0)
        if input_tensor.shape[0] == -1:  # Dynamic batch size
            profile = self.builder.create_optimization_profile()
            input_shape = input_tensor.shape
            
            # Set dynamic batch size range
            min_shape = [1] + list(input_shape[1:])
            opt_shape = [config['max_batch_size']] + list(input_shape[1:])
            max_shape = [config['max_batch_size'] * 2] + list(input_shape[1:])
            
            profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
            builder_config.add_optimization_profile(profile)
            print(f"📏 Dynamic shapes - Min: {min_shape}, Opt: {opt_shape}, Max: {max_shape}")
        
        # Build engine with compatibility handling
        print("🔨 Building TensorRT engine (this may take several minutes)...")
        start_time = time.time()
        
        try:
            # Try newer API first
            if hasattr(self.builder, 'build_serialized_network'):
                serialized_engine = self.builder.build_serialized_network(network, builder_config)
                if serialized_engine is None:
                    print("❌ Failed to build TensorRT engine")
                    return None
                engine_data = serialized_engine
            else:
                # Fallback to older API
                engine = self.builder.build_engine(network, builder_config)
                if engine is None:
                    print("❌ Failed to build TensorRT engine")
                    return None
                engine_data = engine.serialize()
        
        except Exception as e:
            print(f"❌ Engine build failed: {e}")
            return None
        
        build_time = time.time() - start_time
        print(f"✅ Engine built successfully in {build_time:.1f} seconds")
        
        # Serialize and save engine
        print(f"💾 Saving engine to {engine_path}")
        with open(engine_path, 'wb') as f:
            f.write(engine_data)
        
        # Save configuration info
        config_path = engine_path.replace('.engine', '_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        engine_size_mb = os.path.getsize(engine_path) / (1024 * 1024)
        print(f"📊 Engine size: {engine_size_mb:.1f} MB")
        
        return engine_path
    
    def benchmark_engine(self, engine_path, num_iterations=100):
        """Benchmark the TensorRT engine performance"""
        print(f"🏃 Benchmarking engine: {engine_path}")
        
        # Load engine
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(self.logger)
        engine = runtime.deserialize_cuda_engine(engine_data)
        context = engine.create_execution_context()
        
        # Get input/output info
        input_shape = engine.get_binding_shape(0)
        output_shape = engine.get_binding_shape(1)
        
        # Allocate memory
        input_size = trt.volume(input_shape) * engine.max_batch_size
        output_size = trt.volume(output_shape) * engine.max_batch_size
        
        h_input = cuda.pagelocked_empty(input_size, dtype=np.float32)
        h_output = cuda.pagelocked_empty(output_size, dtype=np.float32)
        d_input = cuda.mem_alloc(h_input.nbytes)
        d_output = cuda.mem_alloc(h_output.nbytes)
        
        # Benchmark
        cuda.memcpy_htod(d_input, h_input)
        
        # Warmup
        for _ in range(10):
            context.execute_v2(bindings=[int(d_input), int(d_output)])
        
        # Actual benchmark
        start_time = time.time()
        for _ in range(num_iterations):
            context.execute_v2(bindings=[int(d_input), int(d_output)])
        
        cuda.Context.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_iterations * 1000  # ms
        fps = 1000 / avg_time
        
        print(f"📈 Benchmark Results ({num_iterations} iterations):")
        print(f"   Average inference time: {avg_time:.2f} ms")
        print(f"   Throughput: {fps:.1f} FPS")
        
        return avg_time, fps

def find_onnx_models(models_dir="models"):
    """Find all ONNX models in the specified directory"""
    if not os.path.exists(models_dir):
        print(f"❌ Models directory '{models_dir}' not found!")
        return []
    
    onnx_files = glob.glob(os.path.join(models_dir, "*.onnx"))
    return onnx_files

def convert_all_models(models_dir="models", output_dir=None):
    """Convert all ONNX models to TensorRT engines"""
    if output_dir is None:
        output_dir = models_dir
    
    onnx_models = find_onnx_models(models_dir)
    
    if not onnx_models:
        print("❌ No ONNX models found!")
        return
    
    print(f"📋 Found {len(onnx_models)} ONNX model(s):")
    for i, model in enumerate(onnx_models):
        size_mb = os.path.getsize(model) / (1024 * 1024)
        print(f"   {i+1}. {os.path.basename(model)} ({size_mb:.1f} MB)")
    
    converter = TensorRTConverter()
    successful = 0
    
    for onnx_path in onnx_models:
        print(f"\n{'='*60}")
        print(f"🔄 Converting: {os.path.basename(onnx_path)}")
        
        engine_path = os.path.join(output_dir, 
                                   os.path.basename(onnx_path).replace('.onnx', '.engine'))
        
        try:
            result = converter.build_engine(onnx_path, engine_path)
            if result:
                successful += 1
                # Benchmark the engine
                converter.benchmark_engine(result)
            
        except Exception as e:
            print(f"❌ Failed to convert {os.path.basename(onnx_path)}: {str(e)}")
    
    print(f"\n✅ Successfully converted {successful}/{len(onnx_models)} models")

def main():
    parser = argparse.ArgumentParser(description='Convert ONNX models to TensorRT engines')
    parser.add_argument('--models-dir', type=str, default='models', 
                       help='Directory containing ONNX models')
    parser.add_argument('--output-dir', type=str, help='Output directory for engines')
    parser.add_argument('--model', type=str, help='Convert specific model')
    parser.add_argument('--fp16', action='store_true', help='Force enable FP16')
    parser.add_argument('--int8', action='store_true', help='Force enable INT8')
    parser.add_argument('--batch-size', type=int, default=1, help='Max batch size')
    parser.add_argument('--workspace', type=int, default=4, help='Workspace size in GB')
    parser.add_argument('--benchmark', action='store_true', help='Benchmark converted engines')
    
    args = parser.parse_args()
    
    # Check CUDA availability
    try:
        device_count = cuda.Device.count()
        if device_count == 0:
            print("❌ No CUDA devices found!")
            return
        print(f"🔧 Found {device_count} CUDA device(s)")
    except Exception as e:
        print(f"❌ CUDA initialization failed: {e}")
        return
    
    if args.model:
        # Convert specific model
        try:
            converter = TensorRTConverter()
            config_override = {}
            
            if args.fp16:
                config_override['fp16'] = True
            if args.int8:
                config_override['int8'] = True
            if args.batch_size != 1:
                config_override['max_batch_size'] = args.batch_size
            if args.workspace != 4:
                config_override['max_workspace_size'] = args.workspace * (1024**3)
            
            model_path = os.path.join(args.models_dir, args.model)
            if not os.path.exists(model_path):
                model_path = args.model  # Try as absolute path
                if not os.path.exists(model_path):
                    print(f"❌ Model not found: {args.model}")
                    return
            
            engine_path = converter.build_engine(model_path, config_override=config_override)
            
            if engine_path and args.benchmark:
                converter.benchmark_engine(engine_path)
                
        except Exception as e:
            print(f"❌ Conversion failed: {e}")
            import traceback
            traceback.print_exc()
    
    else:
        # Convert all models
        try:
            convert_all_models(args.models_dir, args.output_dir)
        except Exception as e:
            print(f"❌ Batch conversion failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    # Check TensorRT installation
    try:
        print(f"🔧 TensorRT version: {trt.__version__}")
        device = cuda.Device(0)
        compute_cap = device.compute_capability()
        print(f"🔧 CUDA compute capability: {compute_cap}")
    except Exception as e:
        print(f"❌ TensorRT/CUDA not properly installed: {e}")
        print("💡 Please install: pip install tensorrt pycuda")
        exit(1)
    
    main()