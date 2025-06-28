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
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        """Determine optimal TensorRT configuration for TensorRT 10+ based on model and hardware"""
        
        try:
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
            compute_capability = device.compute_capability()
            
            # TensorRT 10+ optimized configuration
            config = {
                'max_workspace_size': min(4, int(total_memory * 0.5)) * (1024**3),  # Increased workspace for TRT 10+
                'fp16': True,
                'int8': False,  # Start with FP16, enable INT8 later if needed
                'tf32': True,
                'dla_core': None,
                'strict_types': False,  # More flexible for TRT 10+
                'max_batch_size': 1,
                'optimization_level': 5,  # Higher optimization for TRT 10+
                'quantization_flags': [],
                'calibration_dataset_size': 100,
                'enable_plugins': True,
                'verbose': True,
                'sparsity': False,  # TensorRT 10+ sparsity support
                'version_compatible': True,  # Enable version compatibility
                'exclude_lean_runtime': False,  # Include lean runtime
                'builder_optimization_level': 5,  # New in TRT 10+
                'profile_verbosity': 'layer_names_only',  # Detailed profiling
                'tactic_sources': ['CUBLAS', 'CUDNN', 'EDGE_MASK_CONVOLUTIONS']  # TRT 10+ tactics
            }
            
            # Adjust based on GPU architecture and TensorRT 10+ features
            if compute_capability >= (8, 0):  # RTX 30xx/40xx series
                config['fp16'] = True
                config['tf32'] = True
                config['builder_optimization_level'] = 5
                config['enable_all_tactics'] = True
            elif compute_capability >= (7, 5):  # RTX 20xx series
                config['fp16'] = True
                config['tf32'] = True
                config['builder_optimization_level'] = 4
            elif compute_capability >= (7, 0):  # GTX 10xx series
                config['fp16'] = True
                config['tf32'] = False
                config['builder_optimization_level'] = 3
            else:
                # Older GPUs
                config['fp16'] = False
                config['tf32'] = False
                config['builder_optimization_level'] = 2
            
            # Model-specific optimizations for TensorRT 10+
            if 'yolo' in model_name:
                if 'v8' in model_name or 'v11' in model_name:
                    # YOLOv8/v11 optimizations for TRT 10+
                    config['optimization_level'] = 5
                    config['strict_types'] = False
                    config['max_workspace_size'] = 2 * (1024**3)  # 2GB for complex models
                    config['enable_all_tactics'] = True
                elif 'v5' in model_name:
                    # YOLOv5 optimizations
                    config['optimization_level'] = 4
                    config['strict_types'] = False
            
            # Fire/smoke model specific settings
            if 'fire' in model_name or 'smoke' in model_name:
                config['optimization_level'] = 5
                config['fp16'] = True
                config['enable_all_tactics'] = True
                config['profile_verbosity'] = 'detailed'
            
            logger.info(f"🎯 TensorRT 10+ optimized configuration for {gpu_name}:")
            logger.info(f"   Compute Capability: {compute_capability}")
            logger.info(f"   GPU Memory: {total_memory:.1f} GB")
            logger.info(f"   Workspace Size: {config['max_workspace_size'] / (1024**3):.1f} GB")
            logger.info(f"   FP16: {config['fp16']}")
            logger.info(f"   TF32: {config['tf32']}")
            logger.info(f"   Optimization Level: {config['optimization_level']}")
            logger.info(f"   Builder Optimization: {config['builder_optimization_level']}")
            
            return config
            
        except Exception as e:
            logger.error(f"Error getting optimal config: {e}")
            # Return safe fallback config
            return {
                'max_workspace_size': 1 * (1024**3),
                'fp16': True,
                'int8': False,
                'tf32': True,
                'dla_core': None,
                'strict_types': False,
                'max_batch_size': 1,
                'optimization_level': 3,
                'quantization_flags': [],
                'calibration_dataset_size': 100,
                'enable_plugins': True,
                'verbose': True,
                'builder_optimization_level': 3
            }
    
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
        """Build TensorRT engine with TensorRT 10+ optimized configuration"""
        
        try:
            # Create network with explicit batch
            network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            network = self.builder.create_network(network_flags)
            parser = trt.OnnxParser(network, self.logger)
            
            # Parse ONNX model
            logger.info("📖 Parsing ONNX model...")
            with open(onnx_path, 'rb') as model:
                if not parser.parse(model.read()):
                    logger.error("❌ Failed to parse ONNX model")
                    for error in range(parser.num_errors):
                        logger.error(f"   Error {error}: {parser.get_error(error)}")
                    return None
            
            # Create builder configuration
            builder_config = self.builder.create_builder_config()
            
            # TensorRT 10+ memory pool configuration
            try:
                # Use new memory pool API (TensorRT 8.5+)
                builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, config['max_workspace_size'])
                logger.info(f"💾 Workspace memory: {config['max_workspace_size'] / (1024**3):.1f} GB")
            except AttributeError:
                # Fallback to old API
                builder_config.max_workspace_size = config['max_workspace_size']
                logger.warning("⚠️  Using legacy workspace API")
            
            # TensorRT 10+ builder optimization level
            try:
                builder_config.set_builder_optimization_level(config.get('builder_optimization_level', 5))
                logger.info(f"🚀 Builder optimization level: {config.get('builder_optimization_level', 5)}")
            except AttributeError:
                logger.warning("⚠️  Builder optimization level not supported")
            
            # Configure precision modes
            if config.get('fp16', True):
                builder_config.set_flag(trt.BuilderFlag.FP16)
                logger.info("⚡ FP16 precision enabled")
            
            if config.get('tf32', True):
                builder_config.set_flag(trt.BuilderFlag.TF32)
                logger.info("⚡ TF32 precision enabled")
            
            # TensorRT 10+ strict types configuration
            if not config.get('strict_types', False):
                if hasattr(trt.BuilderFlag, 'STRICT_TYPES'):
                    # Allow more flexible type conversions in TRT 10+
                    pass  # Don't set strict types flag
                    logger.info("🔧 Flexible type handling enabled")
            
            # TensorRT 10+ version compatibility
            if config.get('version_compatible', True):
                try:
                    builder_config.set_flag(trt.BuilderFlag.VERSION_COMPATIBLE)
                    logger.info("🔄 Version compatibility enabled")
                except AttributeError:
                    pass
            
            # TensorRT 10+ profiling verbosity
            try:
                if config.get('profile_verbosity') == 'detailed':
                    builder_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
                elif config.get('profile_verbosity') == 'layer_names_only':
                    builder_config.profiling_verbosity = trt.ProfilingVerbosity.LAYER_NAMES_ONLY
                else:
                    builder_config.profiling_verbosity = trt.ProfilingVerbosity.NONE
            except AttributeError:
                pass
            
            # TensorRT 10+ tactic sources
            try:
                if config.get('enable_all_tactics', False):
                    builder_config.set_tactic_sources(1 << int(trt.TacticSource.CUBLAS) |
                                                    1 << int(trt.TacticSource.CUDNN) |
                                                    1 << int(trt.TacticSource.CUBLAS_LT))
                    logger.info("🎯 All tactic sources enabled")
            except AttributeError:
                pass
            
            # Dynamic shapes configuration
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
                logger.info(f"📏 Dynamic shapes - Min: {min_shape}, Opt: {opt_shape}, Max: {max_shape}")
            
            # Build engine
            logger.info("🔨 Building TensorRT engine (this may take several minutes)...")
            start_time = time.time()
            
            # Use TensorRT 10+ build API
            try:
                # Try the newest API first
                serialized_engine = self.builder.build_serialized_network(network, builder_config)
                if serialized_engine is None:
                    logger.error("❌ Failed to build TensorRT engine")
                    return None
                engine_data = serialized_engine
            except Exception as e:
                logger.error(f"❌ Engine build failed: {e}")
                # Try fallback to older API if available
                try:
                    engine = self.builder.build_engine(network, builder_config)
                    if engine is None:
                        logger.error("❌ Failed to build TensorRT engine with fallback API")
                        return None
                    engine_data = engine.serialize()
                    logger.warning("⚠️  Used fallback build API")
                except Exception as fallback_e:
                    logger.error(f"❌ Fallback build also failed: {fallback_e}")
                    return None
            
            build_time = time.time() - start_time
            logger.info(f"✅ Engine built successfully in {build_time:.1f} seconds")
            
            # Save engine
            logger.info(f"💾 Saving engine to {engine_path}")
            os.makedirs(os.path.dirname(engine_path), exist_ok=True)
            with open(engine_path, 'wb') as f:
                f.write(engine_data)
            
            # Save configuration
            config_path = engine_path.replace('.engine', '_config.json')
            # Make config JSON serializable
            json_config = {}
            for key, value in config.items():
                if isinstance(value, (int, float, str, bool, list)) or value is None:
                    json_config[key] = value
                else:
                    json_config[key] = str(value)
            
            with open(config_path, 'w') as f:
                json.dump(json_config, f, indent=2)
            
            # Report results
            engine_size_mb = os.path.getsize(engine_path) / (1024 * 1024)
            logger.info(f"📊 Engine size: {engine_size_mb:.1f} MB")
            logger.info(f"📁 Config saved: {config_path}")
            
            return engine_path
            
        except Exception as e:
            logger.error(f"❌ Engine build failed with error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def benchmark_engine(self, engine_path, num_iterations=100):
        """Benchmark TensorRT engine performance with TensorRT 10+ compatibility"""
        logger.info(f"🏃 Benchmarking engine: {engine_path}")
        
        try:
            # Load engine
            with open(engine_path, 'rb') as f:
                engine_data = f.read()
            
            runtime = trt.Runtime(self.logger)
            engine = runtime.deserialize_cuda_engine(engine_data)
            context = engine.create_execution_context()
            
            # TensorRT 10+ API compatibility for getting tensor info
            try:
                # Try new API first (TensorRT 10+)
                input_names = []
                output_names = []
                
                for i in range(engine.num_io_tensors):
                    name = engine.get_tensor_name(i)
                    if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                        input_names.append(name)
                    else:
                        output_names.append(name)
                
                input_shape = engine.get_tensor_shape(input_names[0])
                output_shape = engine.get_tensor_shape(output_names[0])
                
                logger.info(f"📏 Input shape: {input_shape}")
                logger.info(f"📏 Output shape: {output_shape}")
                
            except AttributeError:
                # Fallback to old API
                logger.warning("⚠️  Using legacy tensor info API")
                input_shape = engine.get_binding_shape(0)
                output_shape = engine.get_binding_shape(1)
                input_names = [engine.get_binding_name(0)]
                output_names = [engine.get_binding_name(1)]
            
            # Calculate tensor sizes
            input_size = np.prod(input_shape)
            output_size = np.prod(output_shape)
            
            # Allocate memory
            h_input = cuda.pagelocked_empty(input_size, dtype=np.float32)
            h_output = cuda.pagelocked_empty(output_size, dtype=np.float32)
            d_input = cuda.mem_alloc(h_input.nbytes)
            d_output = cuda.mem_alloc(h_output.nbytes)
            
            # Initialize input with random data
            h_input[:] = np.random.randn(*h_input.shape).astype(np.float32)
            cuda.memcpy_htod(d_input, h_input)
            
            # Set tensor addresses for TensorRT 10+
            try:
                # Use new API
                context.set_tensor_address(input_names[0], int(d_input))
                context.set_tensor_address(output_names[0], int(d_output))
                use_new_api = True
            except AttributeError:
                # Use old binding API
                use_new_api = False
                bindings = [int(d_input), int(d_output)]
            
            # Warmup runs
            logger.info(f"🔥 Warming up with 10 iterations...")
            for _ in range(10):
                if use_new_api:
                    context.execute_async_v3(cuda.Stream().handle)
                else:
                    context.execute_v2(bindings=bindings)
            
            cuda.Context.synchronize()
            
            # Actual benchmark
            logger.info(f"⏱️  Running benchmark with {num_iterations} iterations...")
            start_time = time.time()
            
            for _ in range(num_iterations):
                if use_new_api:
                    context.execute_async_v3(cuda.Stream().handle)
                else:
                    context.execute_v2(bindings=bindings)
            
            cuda.Context.synchronize()
            end_time = time.time()
            
            # Calculate metrics
            total_time = end_time - start_time
            avg_time = total_time / num_iterations * 1000  # ms
            fps = 1000 / avg_time
            
            logger.info(f"📈 Benchmark Results ({num_iterations} iterations):")
            logger.info(f"   Total time: {total_time:.3f} seconds")
            logger.info(f"   Average inference time: {avg_time:.2f} ms")
            logger.info(f"   Throughput: {fps:.1f} FPS")
            logger.info(f"   Memory usage: Input={h_input.nbytes/1024/1024:.1f}MB, Output={h_output.nbytes/1024/1024:.1f}MB")
            
            # Cleanup
            del h_input, h_output
            d_input.free()
            d_output.free()
            
            return avg_time, fps
            
        except Exception as e:
            logger.error(f"❌ Benchmark failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None, None

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
    parser = argparse.ArgumentParser(description='Convert ONNX models to TensorRT engines with TensorRT 10+ optimization')
    parser.add_argument('--models-dir', type=str, default='models', 
                       help='Directory containing ONNX models')
    parser.add_argument('--output-dir', type=str, help='Output directory for engines')
    parser.add_argument('--model', type=str, help='Convert specific model')
    parser.add_argument('--fp16', action='store_true', help='Force enable FP16')
    parser.add_argument('--int8', action='store_true', help='Force enable INT8')
    parser.add_argument('--tf32', action='store_true', default=True, help='Enable TF32 (default: True)')
    parser.add_argument('--batch-size', type=int, default=1, help='Max batch size')
    parser.add_argument('--workspace', type=int, default=4, help='Workspace size in GB')
    parser.add_argument('--optimization-level', type=int, default=5, help='Builder optimization level (0-5)')
    parser.add_argument('--strict-types', action='store_true', help='Enable strict type constraints')
    parser.add_argument('--version-compatible', action='store_true', default=False, help='Enable version compatibility')
    parser.add_argument('--no-version-compatible', action='store_true', help='Disable version compatibility')
    parser.add_argument('--profile-verbosity', choices=['none', 'layer_names_only', 'detailed'], 
                       default='layer_names_only', help='Profiling verbosity level')
    parser.add_argument('--enable-all-tactics', action='store_true', help='Enable all tactic sources')
    parser.add_argument('--benchmark', action='store_true', help='Benchmark converted engines')
    parser.add_argument('--benchmark-iterations', type=int, default=100, help='Number of benchmark iterations')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Display TensorRT version and system info
    try:
        logger.info(f"🔧 TensorRT version: {trt.__version__}")
        device = cuda.Device(0)
        gpu_name = device.name()
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode()
        compute_cap = device.compute_capability()
        total_memory = device.total_memory() / (1024**3)
        logger.info(f"🎮 GPU: {gpu_name}")
        logger.info(f"📊 Compute Capability: {compute_cap}")
        logger.info(f"💾 GPU Memory: {total_memory:.1f} GB")
    except Exception as e:
        logger.error(f"❌ TensorRT/CUDA not properly installed: {e}")
        logger.error("💡 Please install: pip install tensorrt pycuda")
        return
    
    # Check CUDA availability
    try:
        device_count = cuda.Device.count()
        if device_count == 0:
            logger.error("❌ No CUDA devices found!")
            return
        logger.info(f"🔧 Found {device_count} CUDA device(s)")
    except Exception as e:
        logger.error(f"❌ CUDA initialization failed: {e}")
        return
    
    if args.model:
        # Convert specific model
        try:
            logger.info(f"🚀 Converting specific model: {args.model}")
            converter = TensorRTConverter()
            
            # Build configuration override
            config_override = {
                'builder_optimization_level': args.optimization_level,
                'strict_types': args.strict_types,
                'version_compatible': args.version_compatible,
                'profile_verbosity': args.profile_verbosity,
                'enable_all_tactics': args.enable_all_tactics,
            }
            
            if args.fp16:
                config_override['fp16'] = True
                logger.info("⚡ FP16 forced enabled")
            if args.int8:
                config_override['int8'] = True
                logger.info("⚡ INT8 forced enabled")
            if not args.tf32:
                config_override['tf32'] = False
                logger.info("⚠️  TF32 disabled")
            if args.batch_size != 1:
                config_override['max_batch_size'] = args.batch_size
                logger.info(f"📦 Max batch size: {args.batch_size}")
            if args.workspace != 4:
                config_override['max_workspace_size'] = args.workspace * (1024**3)
                logger.info(f"💾 Workspace size: {args.workspace} GB")
            if args.no_version_compatible:
                config_override['version_compatible'] = False
                logger.info("🚫 Version compatibility disabled")
            
            # Find model path
            model_path = os.path.join(args.models_dir, args.model)
            if not os.path.exists(model_path):
                model_path = args.model  # Try as absolute path
                if not os.path.exists(model_path):
                    logger.error(f"❌ Model not found: {args.model}")
                    return
            
            logger.info(f"📁 Model path: {model_path}")
            
            # Convert model
            engine_path = converter.build_engine(model_path, config_override=config_override)
            
            if engine_path:
                logger.info(f"✅ Conversion successful: {engine_path}")
                
                # Benchmark if requested
                if args.benchmark:
                    logger.info("🏃 Starting benchmark...")
                    avg_time, fps = converter.benchmark_engine(engine_path, args.benchmark_iterations)
                    if avg_time is not None:
                        logger.info(f"🎯 Final Results: {avg_time:.2f}ms avg, {fps:.1f} FPS")
            else:
                logger.error("❌ Conversion failed")
                
        except Exception as e:
            logger.error(f"❌ Conversion failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    else:
        # Convert all models
        try:
            logger.info(f"🔄 Converting all models in directory: {args.models_dir}")
            convert_all_models(args.models_dir, args.output_dir)
        except Exception as e:
            logger.error(f"❌ Batch conversion failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

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