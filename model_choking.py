import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
from PIL import Image, ImageTk
import threading
import time
from ultralytics import YOLO
import numpy as np
from datetime import datetime
import os

class ChokingDetectionSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Choking Detection System")
        self.root.geometry("1200x800")
        
        # Initialize variables
        self.model = None
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.detection_thread = None
        
        # Detection parameters
        self.confidence_threshold = 0.5
        self.model_path = "choking.pt"
        
        # Detection status
        self.choking_detected = False
        self.last_detection_time = None
        self.detection_count = 0
        
        # Setup GUI
        self.setup_gui()
        
        # Try to load model on startup
        self.load_model()
    
    def setup_gui(self):
        """Setup the main GUI components"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Control panel (left side)
        self.setup_control_panel(main_frame)
        
        # Video display (right side)
        self.setup_video_panel(main_frame)
        
        # Status bar (bottom)
        self.setup_status_bar(main_frame)
    
    def setup_control_panel(self, parent):
        """Setup the control panel with buttons and settings"""
        control_frame = ttk.LabelFrame(parent, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Model section
        model_frame = ttk.LabelFrame(control_frame, text="Model", padding="5")
        model_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Button(model_frame, text="Load Model", command=self.browse_model).pack(fill="x", pady=2)
        
        self.model_status = ttk.Label(model_frame, text="No model loaded", foreground="red")
        self.model_status.pack(fill="x", pady=2)
        
        # Camera section
        camera_frame = ttk.LabelFrame(control_frame, text="Camera", padding="5")
        camera_frame.pack(fill="x", pady=(0, 10))
        
        self.start_btn = ttk.Button(camera_frame, text="Start Camera", command=self.start_camera)
        self.start_btn.pack(fill="x", pady=2)
        
        self.stop_btn = ttk.Button(camera_frame, text="Stop Camera", command=self.stop_camera, state="disabled")
        self.stop_btn.pack(fill="x", pady=2)
        
        # Settings section
        settings_frame = ttk.LabelFrame(control_frame, text="Settings", padding="5")
        settings_frame.pack(fill="x", pady=(0, 10))
        
        # Confidence threshold
        ttk.Label(settings_frame, text="Confidence Threshold:").pack(anchor="w")
        self.confidence_var = tk.DoubleVar(value=self.confidence_threshold)
        confidence_scale = ttk.Scale(settings_frame, from_=0.1, to=1.0, variable=self.confidence_var, 
                                   orient="horizontal", command=self.update_confidence)
        confidence_scale.pack(fill="x", pady=2)
        
        self.confidence_label = ttk.Label(settings_frame, text=f"Value: {self.confidence_threshold:.2f}")
        self.confidence_label.pack(anchor="w")
        
        # Detection info section
        info_frame = ttk.LabelFrame(control_frame, text="Detection Info", padding="5")
        info_frame.pack(fill="x", pady=(0, 10))
        
        self.detection_status = ttk.Label(info_frame, text="No Detection", foreground="green", font=("Arial", 12, "bold"))
        self.detection_status.pack(fill="x", pady=2)
        
        self.detection_counter = ttk.Label(info_frame, text="Detections: 0")
        self.detection_counter.pack(fill="x", pady=2)
        
        self.last_detection = ttk.Label(info_frame, text="Last: Never")
        self.last_detection.pack(fill="x", pady=2)
        
        # Alert section
        alert_frame = ttk.LabelFrame(control_frame, text="Alert", padding="5")
        alert_frame.pack(fill="x", pady=(0, 10))
        
        self.alert_btn = ttk.Button(alert_frame, text="Test Alert", command=self.test_alert)
        self.alert_btn.pack(fill="x", pady=2)
        
        self.save_btn = ttk.Button(alert_frame, text="Save Screenshot", command=self.save_screenshot)
        self.save_btn.pack(fill="x", pady=2)
    
    def setup_video_panel(self, parent):
        """Setup the video display panel"""
        video_frame = ttk.LabelFrame(parent, text="Camera Feed", padding="10")
        video_frame.grid(row=0, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Video display
        self.video_label = ttk.Label(video_frame, text="No camera feed", anchor="center")
        self.video_label.pack(expand=True, fill="both")
    
    def setup_status_bar(self, parent):
        """Setup the status bar"""
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(parent, textvariable=self.status_var, relief="sunken")
        status_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
    
    def browse_model(self):
        """Browse and load a YOLO model file"""
        file_path = filedialog.askopenfilename(
            title="Select YOLO Model",
            filetypes=[("PyTorch files", "*.pt"), ("All files", "*.*")]
        )
        if file_path:
            self.model_path = file_path
            self.load_model()
    
    def load_model(self):
        """Load the YOLO model"""
        try:
            if os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
                self.model_status.config(text="Model loaded successfully", foreground="green")
                self.status_var.set(f"Model loaded: {os.path.basename(self.model_path)}")
            else:
                self.model_status.config(text="Model file not found", foreground="red")
                self.status_var.set("Model file not found")
        except Exception as e:
            self.model_status.config(text=f"Error loading model", foreground="red")
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Model Error", f"Failed to load model: {str(e)}")
    
    def start_camera(self):
        """Start the camera and detection"""
        if self.model is None:
            messagebox.showwarning("No Model", "Please load a model first!")
            return
        
        try:
            self.cap = cv2.VideoCapture("rtsp://admin:@2.55.92.197/play1.sdp")  # Use default camera
            if not self.cap.isOpened():
                messagebox.showerror("Camera Error", "Could not open camera!")
                return
            
            self.is_running = True
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            
            # Start detection in separate thread
            self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
            self.detection_thread.start()
            
            self.status_var.set("Camera started - Detection active")
            
        except Exception as e:
            messagebox.showerror("Camera Error", f"Failed to start camera: {str(e)}")
    
    def stop_camera(self):
        """Stop the camera and detection"""
        self.is_running = False
        
        if self.cap:
            self.cap.release()
        
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        
        # Clear video display
        self.video_label.config(image="", text="No camera feed")
        
        self.status_var.set("Camera stopped")
    
    def detection_loop(self):
        """Main detection loop running in separate thread"""
        while self.is_running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self.current_frame = frame.copy()
                    
                    # Run detection
                    detections = self.model(frame, conf=self.confidence_threshold)
                    
                    # Process detections
                    self.process_detections(frame, detections)
                    
                    # Update display
                    self.update_video_display(frame)
                
            time.sleep(0.033)  # ~30 FPS
    
    def process_detections(self, frame, detections):
        """Process YOLO detections and update status"""
        choking_detected = False
        
        for detection in detections:
            boxes = detection.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    # Get class name (assuming your model has class names)
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Check if this is a choking detection
                    # You may need to adjust this based on your model's class names
                    class_names = self.model.names if hasattr(self.model, 'names') else {0: 'choking'}
                    class_name = class_names.get(class_id, 'unknown')
                    
                    if 'chok' in class_name.lower() or class_id == 0:  # Adjust as needed
                        choking_detected = True
                        
                        # Draw bounding box
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, f'{class_name}: {confidence:.2f}', 
                                  (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Update detection status
        self.update_detection_status(choking_detected)
    
    def update_detection_status(self, choking_detected):
        """Update the detection status in the GUI"""
        if choking_detected and not self.choking_detected:
            # New choking detection
            self.choking_detected = True
            self.detection_count += 1
            self.last_detection_time = datetime.now()
            
            # Update GUI in main thread
            self.root.after(0, self.update_detection_gui, True)
            
            # Trigger alert
            self.root.after(0, self.trigger_alert)
            
        elif not choking_detected and self.choking_detected:
            # Choking stopped
            self.choking_detected = False
            self.root.after(0, self.update_detection_gui, False)
    
    def update_detection_gui(self, choking_detected):
        """Update detection GUI elements (called from main thread)"""
        if choking_detected:
            self.detection_status.config(text="CHOKING DETECTED!", foreground="red")
        else:
            self.detection_status.config(text="No Detection", foreground="green")
        
        self.detection_counter.config(text=f"Detections: {self.detection_count}")
        
        if self.last_detection_time:
            time_str = self.last_detection_time.strftime("%H:%M:%S")
            self.last_detection.config(text=f"Last: {time_str}")
    
    def update_video_display(self, frame):
        """Update the video display with the current frame"""
        # Convert frame to RGB and resize for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize frame to fit display
        height, width = frame_rgb.shape[:2]
        max_width, max_height = 640, 480
        
        if width > max_width or height > max_height:
            scale = min(max_width/width, max_height/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
        
        # Convert to PhotoImage and update display
        image = Image.fromarray(frame_rgb)
        photo = ImageTk.PhotoImage(image)
        
        # Update in main thread
        self.root.after(0, self._update_video_label, photo)
    
    def _update_video_label(self, photo):
        """Update video label (called from main thread)"""
        self.video_label.config(image=photo, text="")
        self.video_label.photo = photo  # Keep a reference
    
    def update_confidence(self, value):
        """Update confidence threshold"""
        self.confidence_threshold = float(value)
        self.confidence_label.config(text=f"Value: {self.confidence_threshold:.2f}")
    
    def trigger_alert(self):
        """Trigger alert when choking is detected"""
        # Flash the window
        self.root.bell()
        
        # You can add more alert mechanisms here:
        # - Play sound
        # - Send email/SMS
        # - Log to file
        # - etc.
        
        messagebox.showwarning("ALERT", "CHOKING DETECTED!\n\nImmediate assistance may be required!")
    
    def test_alert(self):
        """Test the alert system"""
        self.trigger_alert()
    
    def save_screenshot(self):
        """Save current frame as screenshot"""
        if self.current_frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"choking_detection_{timestamp}.jpg"
            
            filepath = filedialog.asksaveasfilename(
                defaultextension=".jpg",
                filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")],
                initialvalue=filename
            )
            
            if filepath:
                cv2.imwrite(filepath, self.current_frame)
                messagebox.showinfo("Screenshot Saved", f"Screenshot saved as: {filepath}")
        else:
            messagebox.showwarning("No Frame", "No frame available to save!")
    
    def on_closing(self):
        """Handle application closing"""
        self.stop_camera()
        self.root.destroy()

def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = ChokingDetectionSystem(root)
    
    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Start the GUI
    root.mainloop()

if __name__ == "__main__":
    main()