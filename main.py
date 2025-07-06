import multiprocessing
import cv2
from fire_detector_class import FireDetector
# Uncomment and add other detectors when needed
from choking_detector_class import ChokingDetector
from fall_detector_class import FallDetector
from violence_detector_class import ViolenceDetector

def display_frame(detector_config):
    detector_class = None
    cap = None
    
    try:
        detector_type = detector_config['detector_type']
        
        if detector_type == 'fire':
            detector_class = FireDetector(
                detector_config['rtsp_url'],
                detector_config['video_file'],
                detector_config['use_rtsp']
            )
        elif detector_type == 'fall':
            detector_class = FallDetector(
                detector_config['rtsp_url'],
                detector_config['video_file'],
                detector_config['use_rtsp']
            )
        elif detector_type == 'choking':
            detector_class = ChokingDetector(
                detector_config['rtsp_url'],
                detector_config['video_file'],
                detector_config['use_rtsp']
            )
        elif detector_type == 'violence':
            detector_class = ViolenceDetector(
                detector_config['rtsp_url'],
                detector_config['video_file'],
                detector_config['use_rtsp']
            )
        else:
            print(f"[ERROR] Unknown detector type: {detector_type}")
            return
        
        cap = cv2.VideoCapture(detector_class.source)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {detector_class.source}")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"[INFO] {detector_type.capitalize()} Detector started - Video: {width}x{height} @ {fps}fps")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                # If it's a video file, loop it
                if not detector_config['use_rtsp']:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    print(f"[INFO] {detector_type.capitalize()} Detector: End of stream reached")
                    break

            # Process frame using the detector class
            frame = detector_class.process_frame(frame, fps)

            # Show the frame with OpenCV (with detector type in window name)
            window_name = f"{detector_type.capitalize()} Detection - PID:{multiprocessing.current_process().pid}"
            cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print(f"[INFO] {detector_type.capitalize()} Detector: Quit key pressed")
                break
                
            frame_count += 1
            if frame_count % 300 == 0:  # Log every 300 frames (10 seconds at 30fps)
                print(f"[INFO] {detector_type.capitalize()} Detector processed {frame_count} frames")
                
    except KeyboardInterrupt:
        print(f"[INFO] {detector_type.capitalize()} Detector: Stopping detection...")
    except Exception as e:
        print(f"[ERROR] Error in {detector_type.capitalize()} Detector: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        # Clean up detector resources
        if detector_class is not None:
            detector_class.cleanup()
        print(f"[INFO] {detector_config.get('detector_type', 'Unknown')} Detector cleanup completed")

def run_detectors(rtsp_url, video_file, use_rtsp=False, enabled_detectors=None):
    if enabled_detectors is None:
        enabled_detectors = ['fire', 'fall', 'choking', 'violence']  # Default to fire and fall detectors
    
    # Create detector configurations (don't initialize the detector objects here)
    detector_configs = []
    
    for detector_type in enabled_detectors:
        detector_config = {
            'detector_type': detector_type,
            'rtsp_url': rtsp_url,
            'video_file': video_file,
            'use_rtsp': use_rtsp
        }
        detector_configs.append(detector_config)
        print(f"[INFO] Added {detector_type.capitalize()} Detector to configuration")

    if not detector_configs:
        print("[ERROR] No detectors configured!")
        return

    # Create a process for each detector configuration
    processes = []
    for detector_config in detector_configs:
        process = multiprocessing.Process(
            target=display_frame, 
            args=(detector_config,),
            name=f"{detector_config['detector_type'].capitalize()}DetectorProcess"
        )
        processes.append(process)
        process.start()
        print(f"[INFO] Started {detector_config['detector_type'].capitalize()} Detector process (PID: {process.pid})")

    try:
        print(f"[INFO] Running {len(processes)} detector processes. Press Ctrl+C to stop or 'q' in any window.")
        # Wait for all processes to complete
        for process in processes:
            process.join()
    except KeyboardInterrupt:
        print("[INFO] Terminating all processes...")
        for process in processes:
            if process.is_alive():
                print(f"[INFO] Terminating {process.name}...")
                process.terminate()
        
        # Wait for processes to terminate gracefully
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                print(f"[WARNING] Force killing {process.name}...")
                process.kill()
                process.join()
    
    print("[INFO] All detector processes have been stopped.")

if __name__ == "__main__":
    rtsp_url = "rtsp://admin:@2.55.92.197/play1.sdp"  # Example RTSP stream URL
    video_file = "video.mp4" 
    use_rtsp = False 
    # enabled_detectors = ['fire', 'fall', 'choking', 'violence']  # Enable both fire and 
    
    # You can also enable specific detectors based on your needs:
    enabled_detectors = ['violence']  # Only fire detection
    # enabled_detectors = ['fall']  # Only fall detection
    # enabled_detectors = ['fire', 'fall', 'choking', 'violence']  # All detectors
    
    run_detectors(rtsp_url, video_file, use_rtsp, enabled_detectors)