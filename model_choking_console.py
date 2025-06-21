#!/usr/bin/env python3
"""
Choking Detection System - Console Version
Ubuntu headless system compatible choking detection script
Automatically detects choking events and saves annotated images
"""

import cv2
import threading
import time
from ultralytics import YOLO
import numpy as np
from datetime import datetime
import os
import sys
import argparse
import signal
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('choking_detection.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ChokingDetectionSystemConsole:
    def __init__(self, model_path="models/choking.pt", camera_source=0, confidence_threshold=0.87, output_dir="choking_detections"):
        """
        Initialize choking detection system for console operation
        
        Args:
            model_path: Path to the YOLO model file
            camera_source: Camera source (0 for default, or RTSP URL)
            confidence_threshold: Detection confidence threshold
            output_dir: Directory to save detection images
        """
        # Initialize variables
        self.model_path = model_path
        self.camera_source = camera_source
        self.confidence_threshold = confidence_threshold
        self.output_dir = output_dir
        
        self.model = None
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.detection_thread = None
        
        # Detection status
        self.choking_detected = False
        self.last_detection_time = None
        self.detection_count = 0
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        logger.info("Choking Detection System (Console) initialized")
    
    def signal_handler(self, sig, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {sig}, shutting down gracefully...")
        self.stop_detection()
        sys.exit(0)
    
    def load_model(self):
        """Load the YOLO model"""
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                return False
            
            logger.info(f"Loading YOLO model from: {self.model_path}")
            self.model = YOLO(self.model_path)
            logger.info("Model loaded successfully")
            
            # Print model info
            if hasattr(self.model, 'names'):
                logger.info(f"Model classes: {self.model.names}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False
    
    def start_camera(self):
        """Start the camera with enhanced RTSP support"""
        try:
            logger.info(f"Opening camera source: {self.camera_source}")
            
            # Configure OpenCV backend for RTSP streams
            if isinstance(self.camera_source, str) and "rtsp://" in self.camera_source:
                logger.info("Detected RTSP stream, using optimized settings")
                
                # Try different backends for RTSP
                backends = [cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER, cv2.CAP_ANY]
                
                for backend in backends:
                    try:
                        logger.info(f"Trying backend: {backend}")
                        self.cap = cv2.VideoCapture(self.camera_source, backend)
                        
                        if self.cap.isOpened():
                            # Set RTSP-specific properties for better H.264 handling
                            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize delay
                            # Remove FOURCC setting to let OpenCV auto-detect
                            # Additional properties to handle H.264 stream better
                            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                            self.cap.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS for stability
                            logger.info(f"Successfully opened RTSP stream with backend: {backend}")
                            break
                        else:
                            self.cap.release()
                    except Exception as e:
                        logger.warning(f"Backend {backend} failed: {str(e)}")
                        continue
                else:
                    logger.error("All RTSP backends failed, trying basic connection")
                    self.cap = cv2.VideoCapture(self.camera_source)
            else:
                # Regular camera (USB, etc.)
                self.cap = cv2.VideoCapture(self.camera_source)
            
            if not self.cap.isOpened():
                logger.error("Could not open camera!")
                return False
            
            # Test frame reading
            logger.info("Testing frame capture...")
            for attempt in range(5):
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    logger.info(f"Frame test successful on attempt {attempt + 1}")
                    break
                else:
                    logger.warning(f"Frame test failed on attempt {attempt + 1}")
                    time.sleep(1)
            else:
                logger.error("Failed to read test frames from camera")
                return False
            
            # Set camera properties for better performance (only for non-RTSP)
            if not (isinstance(self.camera_source, str) and "rtsp://" in self.camera_source):
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Get actual camera properties
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Camera opened successfully - Resolution: {width}x{height}, FPS: {fps}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start camera: {str(e)}")
            return False
    
    def stop_camera(self):
        """Stop the camera"""
        if self.cap:
            self.cap.release()
            logger.info("Camera stopped")
    
    def start_detection(self):
        """Start the detection system"""
        if not self.load_model():
            return False
        
        if not self.start_camera():
            return False
        
        self.is_running = True
        logger.info("=== Starting Choking Detection ===")
        logger.info(f"Confidence threshold: {self.confidence_threshold}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("Press Ctrl+C to stop detection")
        
        # Start detection in separate thread
        self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.detection_thread.start()
        
        return True
    
    def stop_detection(self):
        """Stop the detection system"""
        logger.info("Stopping detection system...")
        self.is_running = False
        self.stop_camera()
        
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=5)
        
        logger.info("Detection system stopped")
    
    def detection_loop(self):
        """Main detection loop with enhanced error handling"""
        frame_count = 0
        start_time = time.time()
        consecutive_failures = 0
        max_failures = 10
        
        while self.is_running:
            try:
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret and frame is not None and frame.size > 0:
                        # Reset failure counter on successful read
                        consecutive_failures = 0
                        
                        self.current_frame = frame.copy()
                        frame_count += 1
                        
                        # Run detection with error handling for corrupted frames
                        try:
                            detections = self.model(frame, conf=self.confidence_threshold, verbose=False)
                        except Exception as detection_error:
                            logger.warning(f"Detection failed on frame {frame_count}: {str(detection_error)}")
                            continue
                        
                        # Process detections
                        annotated_frame = self.process_detections(frame, detections)
                        
                        # Print status every 100 frames
                        if frame_count % 100 == 0:
                            elapsed = time.time() - start_time
                            fps = frame_count / elapsed
                            logger.info(f"Processed {frame_count} frames, FPS: {fps:.1f}, Detections: {self.detection_count}")
                        
                        # Update current frame with annotations
                        self.current_frame = annotated_frame
                    else:
                        consecutive_failures += 1
                        logger.warning(f"Failed to read frame from camera (failure {consecutive_failures}/{max_failures})")
                        
                        # Try to reconnect if too many failures
                        if consecutive_failures >= max_failures:
                            logger.error("Too many consecutive failures, attempting to reconnect...")
                            self.reconnect_camera()
                            consecutive_failures = 0
                        
                        time.sleep(0.1)
                else:
                    logger.error("Camera not available")
                    # Try to reconnect
                    if not self.reconnect_camera():
                        logger.error("Failed to reconnect camera, stopping detection")
                        break
                    
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                logger.error(f"Error in detection loop: {str(e)}")
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    logger.error("Too many errors, attempting camera reconnection...")
                    self.reconnect_camera()
                    consecutive_failures = 0
                time.sleep(1)
    
    def reconnect_camera(self):
        """Attempt to reconnect to the camera"""
        try:
            logger.info("Attempting to reconnect camera...")
            if self.cap:
                self.cap.release()
            
            time.sleep(2)  # Wait before reconnecting
            return self.start_camera()
            
        except Exception as e:
            logger.error(f"Failed to reconnect camera: {str(e)}")
            return False
    
    def process_detections(self, frame, detections):
        """Process YOLO detections and return annotated frame"""
        choking_detected = False
        annotated_frame = frame.copy()
        
        try:
            for detection in detections:
                boxes = detection.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        # Get detection details
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Get class name
                        class_names = self.model.names if hasattr(self.model, 'names') else {0: 'choking'}
                        class_name = class_names.get(class_id, 'unknown')
                        
                        # Check if this is a choking detection
                        if 'chok' in class_name.lower() or class_id == 0:
                            choking_detected = True
                            
                            # Draw bounding box (red for choking)
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            
                            # Draw label background
                            label = f'{class_name}: {confidence:.2f}'
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                                        (x1 + label_size[0], y1), (0, 0, 255), -1)
                            
                            # Draw label text
                            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            
                            # Add detection info
                            logger.info(f"CHOKING DETECTED! Confidence: {confidence:.2f}, Box: ({x1},{y1})-({x2},{y2})")
            
            # Add timestamp and status to frame
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(annotated_frame, timestamp, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if choking_detected:
                cv2.putText(annotated_frame, "CHOKING DETECTED!", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            else:
                cv2.putText(annotated_frame, "Monitoring...", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Update detection status
            self.update_detection_status(choking_detected, annotated_frame)
            
        except Exception as e:
            logger.error(f"Error processing detections: {str(e)}")
        
        return annotated_frame
    
    def update_detection_status(self, choking_detected, annotated_frame):
        """Update detection status and trigger alerts"""
        if choking_detected and not self.choking_detected:
            # New choking detection
            self.choking_detected = True
            self.detection_count += 1
            self.last_detection_time = datetime.now()
            
            logger.warning(f"🚨 CHOKING ALERT #{self.detection_count} - {self.last_detection_time.strftime('%H:%M:%S')}")
            
            # Trigger alert and save image
            self.trigger_alert(annotated_frame)
            
        elif not choking_detected and self.choking_detected:
            # Choking stopped
            self.choking_detected = False
            logger.info("✅ Choking detection cleared")
    
    def trigger_alert(self, annotated_frame):
        """Trigger alert when choking is detected"""
        try:
            # Save annotated image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
            filename = f"choking_alert_{timestamp}.jpg"
            filepath = os.path.join(self.output_dir, filename)
            
            # Add alert overlay to image
            alert_frame = annotated_frame.copy()
            
            # Add red border
            h, w = alert_frame.shape[:2]
            cv2.rectangle(alert_frame, (0, 0), (w-1, h-1), (0, 0, 255), 10)
            
            # Add large alert text
            cv2.putText(alert_frame, "EMERGENCY ALERT", (w//2 - 150, h//2 - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
            cv2.putText(alert_frame, "CHOKING DETECTED", (w//2 - 150, h//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.putText(alert_frame, f"Alert #{self.detection_count}", (w//2 - 80, h//2 + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            
            # Save the alert image
            cv2.imwrite(filepath, alert_frame)
            logger.warning(f"🖼️  Alert image saved: {filepath}")
            
            # Also save the original annotated frame
            original_filename = f"choking_detection_{timestamp}.jpg"
            original_filepath = os.path.join(self.output_dir, original_filename)
            cv2.imwrite(original_filepath, annotated_frame)
            
            # Print alert to console
            print("\n" + "="*60)
            print("🚨 CHOKING DETECTION ALERT 🚨")
            print(f"Time: {self.last_detection_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Alert Count: {self.detection_count}")
            print(f"Image Saved: {filepath}")
            print("="*60 + "\n")
            
            # You can add additional alert mechanisms here:
            # - Send webhook/API call
            # - Send email/SMS
            # - Write to database
            # - Send notification to monitoring system
            
        except Exception as e:
            logger.error(f"Error triggering alert: {str(e)}")
    
    def run(self):
        """Run the detection system"""
        try:
            if not self.start_detection():
                logger.error("Failed to start detection system")
                return False
            
            # Keep the main thread alive
            while self.is_running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
        finally:
            self.stop_detection()
        
        return True
    
    def print_summary(self):
        """Print detection summary"""
        print("\n" + "="*50)
        print("DETECTION SUMMARY")
        print("="*50)
        print(f"Total Detections: {self.detection_count}")
        if self.last_detection_time:
            print(f"Last Detection: {self.last_detection_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Output Directory: {self.output_dir}")
        print(f"Saved Images: {len([f for f in os.listdir(self.output_dir) if f.endswith('.jpg')])}")
        print("="*50)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Choking Detection System - Console Version")
    parser.add_argument("--model", "-m", default="models/choking.pt", 
                       help="Path to YOLO model file (default: models/choking.pt)")
    parser.add_argument("--camera", "-c", default="rtsp://admin:@2.55.92.197/play1.sdp", 
                       help="Camera source: 0 for default camera, or RTSP URL (default: 0)")
    parser.add_argument("--confidence", "-conf", type=float, default=0.5, 
                       help="Detection confidence threshold (default: 0.87)")
    parser.add_argument("--output", "-o", default="choking_detections", 
                       help="Output directory for detection images (default: choking_detections)")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Convert camera argument
    camera_source = args.camera
    if isinstance(args.camera, str) and args.camera.isdigit():
        camera_source = int(args.camera)
    
    print("="*60)
    print("🔍 CHOKING DETECTION SYSTEM - CONSOLE VERSION")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Camera: {camera_source}")
    print(f"Confidence: {args.confidence}")
    print(f"Output: {args.output}")
    print("="*60)
    
    # Create and run detection system
    detector = ChokingDetectionSystemConsole(
        model_path=args.model,
        camera_source=camera_source,
        confidence_threshold=args.confidence,
        output_dir=args.output
    )
    
    try:
        success = detector.run()
        detector.print_summary()
        
        if success:
            print("Detection system completed successfully")
            return 0
        else:
            print("Detection system failed")
            return 1
            
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())