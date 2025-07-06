import cv2
import numpy as np
import pickle
import os
import time
from datetime import datetime
import tensorrt as trt
import pycuda.driver as cuda
import subprocess
# Do NOT import pycuda.autoinit here - we'll manage CUDA context manually

class ChokingDetector:
    def __init__(self, rtsp_url, video_file, use_rtsp=False):
        # Parameters for RTSP URL and video file
        self.rtsp_url = rtsp_url
        self.video_file = video_file
        self.use_rtsp = use_rtsp
        self.source = self.rtsp_url if self.use_rtsp else self.video_file
        self.output_dir = "./media/detection_clips"
        self.engine_path = "./models/choking.engine"
        self.label_binarizer_path = "./models/choking.pickle"
        self.image_size = 128
        self.choking_label = "Choking"
        self.clip_duration_seconds = 20
        self.cooldown_seconds = 5
        self.last_clip_saved_time = 0
        self.thumbnail_saved = False
        self.frame_buffer = []
        
        # Initialize CUDA context and TensorRT components
        self.cuda_context = None
        self.engine = None
        self.context = None
        self.d_input = None
        self.d_output = None
        self.bindings = None
        self.host_input = None
        self.host_output = None
        self.lb = None
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize components
        self._initialize_components()

    def _check_cuda_availability(self):
        """Check if CUDA is available and properly configured."""
        try:
            # Check if nvidia-smi is available
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError("nvidia-smi not found or failed to run")
            
            print("[INFO] nvidia-smi output:")
            print(result.stdout)
            return True
            
        except ImportError:
            raise RuntimeError("PyCUDA not properly installed")
        except FileNotFoundError:
            raise RuntimeError("nvidia-smi not found. NVIDIA drivers may not be installed.")
        except Exception as e:
            raise RuntimeError(f"CUDA availability check failed: {e}")

    def _initialize_components(self):
        """Initialize all CUDA and TensorRT components in the current process."""
        try:
            # Check CUDA availability first
            self._check_cuda_availability()
            
            # Initialize CUDA in this process (each process needs its own CUDA context)
            try:
                # Check if CUDA is already initialized
                try:
                    cuda.Device.count()
                    print("[INFO] CUDA already initialized")
                except cuda.LogicError:
                    # CUDA not initialized, initialize it
                    cuda.init()
                    print("[INFO] CUDA initialized successfully")
            except Exception as cuda_init_error:
                print(f"[ERROR] CUDA initialization failed: {cuda_init_error}")
                raise
            
            # Check if we have any CUDA devices
            device_count = cuda.Device.count()
            if device_count == 0:
                raise RuntimeError("No CUDA devices found")
            
            print(f"[INFO] Found {device_count} CUDA device(s)")
            
            # Get device info
            device = cuda.Device(0)
            print(f"[INFO] Using device: {device.name()}")
            print(f"[INFO] Device compute capability: {device.compute_capability()}")
            
            # Create CUDA context
            self.cuda_context = device.make_context()
            print("[INFO] CUDA context created successfully")
            
            # Load label binarizer
            with open(self.label_binarizer_path, "rb") as f:
                self.lb = pickle.load(f)
            print("[INFO] Label binarizer loaded successfully")

            # Load TensorRT engine
            self.engine = self.load_engine(self.engine_path)
            self.context = self.engine.create_execution_context()
            print("[INFO] TensorRT engine loaded successfully")

            # Prepare input/output bindings and memory allocations
            self.prepare_memory()
            
        except Exception as e:
            print(f"[ERROR] Failed to initialize ChokingDetector: {e}")
            import traceback
            traceback.print_exc()
            raise

    def load_engine(self, path):
        """Load the TensorRT engine from the specified path."""
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"TensorRT engine file not found: {path}")
                
            with open(path, "rb") as f, trt.Runtime(trt.Logger(trt.Logger.INFO)) as runtime:
                return runtime.deserialize_cuda_engine(f.read())
        except Exception as e:
            print(f"[ERROR] Failed to load TensorRT engine: {e}")
            raise

    def prepare_memory(self):
        """Allocate memory and prepare input/output bindings."""
        try:
            input_name = self.get_input_name()
            output_name = self.get_output_name()

            input_shape = tuple(self.engine.get_tensor_shape(input_name))
            output_shape = tuple(self.engine.get_tensor_shape(output_name))

            print(f"[INFO] Input shape: {input_shape}")
            print(f"[INFO] Output shape: {output_shape}")

            input_size = np.prod(input_shape) * np.dtype(np.float32).itemsize
            output_size = np.prod(output_shape) * np.dtype(np.float32).itemsize

            # Allocate memory for inputs and outputs
            self.d_input = cuda.mem_alloc(int(input_size))
            self.d_output = cuda.mem_alloc(int(output_size))
            self.bindings = [int(self.d_input), int(self.d_output)]
            self.host_input = np.empty(input_shape, dtype=np.float32)
            self.host_output = np.empty(output_shape, dtype=np.float32)
            
            print(f"[INFO] Memory allocation successful")
            
        except Exception as e:
            print(f"[ERROR] Failed to prepare memory: {e}")
            raise

    def get_input_name(self):
        """Get the input tensor name from the TensorRT engine."""
        try:
            input_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
                          if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT]
            if not input_names:
                raise RuntimeError("No input tensors found in engine")
            return input_names[0]
        except Exception as e:
            print(f"[ERROR] Failed to get input tensor name: {e}")
            raise

    def get_output_name(self):
        """Get the output tensor name from the TensorRT engine."""
        try:
            output_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
                           if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]
            if not output_names:
                raise RuntimeError("No output tensors found in engine")
            return output_names[0]
        except Exception as e:
            print(f"[ERROR] Failed to get output tensor name: {e}")
            raise

    def process_frame(self, frame, fps):
        try:
            # Check if initialization was successful
            if self.host_input is None or self.lb is None:
                print("[ERROR] ChokingDetector not properly initialized")
                return frame
            
            # Ensure CUDA context is active
            if self.cuda_context:
                self.cuda_context.push()
            
            # Preprocess frame: resize and normalize it for inference
            resized = cv2.resize(frame, (self.image_size, self.image_size))
            normalized = resized.astype(np.float32) / 255.0
            
            # Check if host_input has the right shape
            if self.host_input.shape[0] == 1:
                self.host_input[0] = normalized
            else:
                # If batch size is different, handle accordingly
                self.host_input = normalized.reshape(self.host_input.shape)

            # Run inference
            cuda.memcpy_htod(self.d_input, self.host_input)
            self.context.execute_v2(self.bindings)
            cuda.memcpy_dtoh(self.host_output, self.d_output)

            # Get predictions and process detection result
            preds = self.host_output[0] if self.host_output.ndim > 1 else self.host_output
            pred_idx = np.argmax(preds)
            label = self.lb.classes_[pred_idx]
            confidence = float(preds[pred_idx])
            current_time = time.time()
            
            self.frame_buffer.append(frame)
            if len(self.frame_buffer) > fps * self.cooldown_seconds:
                self.frame_buffer.pop(0)

            if label == self.choking_label and confidence >= 0.98:
                self.save_clip(frame, current_time)
                print(f"[INFO] Choking detected with confidence: {confidence:.2f}")
                
            # Draw detection info on frame
            color = (0, 0, 255) if label == self.choking_label else (0, 255, 0)
            cv2.putText(frame, f"Detection: {label}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Pop CUDA context
            if self.cuda_context:
                self.cuda_context.pop()
                
        except Exception as e:
            print(f"[ERROR] Error processing frame: {e}")
            import traceback
            traceback.print_exc()
            # Pop CUDA context in case of error
            if self.cuda_context:
                try:
                    self.cuda_context.pop()
                except:
                    pass
        
        return frame
    
    def convert_to_web_compatible(self, video_path):
        """
        Converts a video to a web-compatible format using FFmpeg if available
        """
        try:
            # Check if ffmpeg is available
            result = subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            ffmpeg_available = result.returncode == 0
        except:
            ffmpeg_available = False
            return video_path
        
        if not ffmpeg_available:
            return video_path
        
        # Create new filename for the web-compatible video
        base_dir = os.path.dirname(video_path)
        filename = os.path.basename(video_path)
        name, ext = os.path.splitext(filename)
        web_path = os.path.join(base_dir, f"{name}_web.mp4")
        
        try:
            # Use FFmpeg to convert the video to H.264 in MP4 container (web compatible)
            command = [
                'ffmpeg',
                '-i', video_path,                # Input file
                '-c:v', 'libx264',               # H.264 codec
                '-preset', 'fast',               # Encoding speed/compression tradeoff
                '-crf', '23',                    # Quality (lower = better)
                '-pix_fmt', 'yuv420p',           # Pixel format for compatibility
                '-y',                            # Overwrite output file if it exists
                web_path                         # Output file
            ]
            
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            if result.returncode == 0 and os.path.exists(web_path):
                print(f"Successfully converted video to web format: {web_path}")
                return web_path
            else:
                print(f"FFmpeg conversion failed: {result.stderr.decode()}")
                return video_path
                
        except Exception as e:
            print(f"Error during FFmpeg conversion: {str(e)}")
            return video_path

    def save_clip(self, frame, current_time):
        try:
            if not self.thumbnail_saved:
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                thumbnail_path = os.path.join(self.output_dir, f"choking_thumbnail_{timestamp_str}.jpg")
                cv2.imwrite(thumbnail_path, frame)
                self.thumbnail_saved = True
                print(f"[INFO] Saved thumbnail: {thumbnail_path}")

            # Check cooldown before saving clip
            if current_time - self.last_clip_saved_time > self.cooldown_seconds:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(self.output_dir, f"choking_{timestamp}.mp4")

                # Open video writer to save the clip
                fps = 30  # Assuming the FPS is 30, adjust as needed
                width = frame.shape[1]
                height = frame.shape[0]
                clip_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

                if not clip_writer.isOpened():
                    print(f"[ERROR] Cannot open video writer for {output_path}")
                    return

                # Write frame to clip
                for buf_frame in self.frame_buffer:
                    clip_writer.write(buf_frame)
                clip_writer.release()
                self.frame_buffer.clear()  # Clear the buffer after saving
                self.last_clip_saved_time = current_time
                self.convert_to_web_compatible(output_path)
                os.remove(output_path) 
                print(f"[INFO] Saved clip to {output_path}")
                
        except Exception as e:
            print(f"[ERROR] Error saving clip: {e}")

    def cleanup(self):
        """Clean up resources when done."""
        try:
            if self.d_input:
                self.d_input.free()
            if self.d_output:
                self.d_output.free()
            if self.cuda_context:
                self.cuda_context.pop()
                self.cuda_context.detach()
            print(f"[INFO] ChokingDetector cleanup completed for process {os.getpid()}")
        except Exception as e:
            print(f"[ERROR] Error during cleanup: {e}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass