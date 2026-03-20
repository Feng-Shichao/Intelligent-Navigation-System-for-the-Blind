import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time
import os
import sys
import subprocess

# Import audio guidance module
from audio_guidance import load_audio_files, handle_multiple_people

def generate_colors(num_classes):
    """Generate a list of distinct colors for visualization."""
    np.random.seed(42)  # For reproducibility
    colors = []
    for _ in range(num_classes):
        # Generate random BGR values
        color = (
            np.random.randint(0, 255),  # B
            np.random.randint(0, 255),  # G
            np.random.randint(0, 255)   # R
        )
        colors.append(color)
    return colors

def draw_segmentation(image, masks, boxes, class_names, colors):
    """Draw segmentation masks on the image."""
    segmentation_img = image.copy()
    
    if masks is None:
        return segmentation_img
    
    for i, mask in enumerate(masks):
        # Make sure we have a valid image size
        if segmentation_img.shape[0] <= 0 or segmentation_img.shape[1] <= 0:
            print(f"Invalid image dimensions: {segmentation_img.shape}")
            continue
            
        # Get class ID for this mask
        cls_id = int(boxes[i].cls[0])
        color = colors[cls_id % len(colors)]
        
        try:
            # Convert mask tensor to numpy array
            mask_array = mask.data.cpu().numpy().astype(np.uint8)
            
            # Check if mask has valid dimensions
            if mask_array.size == 0:
                print("Empty mask array, skipping")
                continue
                
            # Get target dimensions, ensure they are positive integers
            target_h, target_w = segmentation_img.shape[0], segmentation_img.shape[1]
            
            # Resize mask to match image dimensions
            if target_h > 0 and target_w > 0:
                mask_resized = cv2.resize(
                    mask_array,
                    (target_w, target_h),
                    interpolation=cv2.INTER_NEAREST
                )
                
                # Create a colored mask
                colored_mask = np.zeros_like(segmentation_img, dtype=np.uint8)
                mask_color = color  # No alpha needed for addWeighted
                
                for c in range(3):
                    colored_mask[:, :, c] = np.where(mask_resized == 1, mask_color[c], 0)
                
                # Overlay the colored mask on the image with transparency
                cv2.addWeighted(segmentation_img, 1, colored_mask, 0.5, 0, segmentation_img)
        except Exception as e:
            print(f"Error processing mask: {e}")
            continue
    
    return segmentation_img

def check_camera_indices():
    """Check which camera indices are available on the system."""
    available_cameras = []
    max_to_check = 10  # Check cameras 0-9
    
    for i in range(max_to_check):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cameras.append(i)
                print(f"Camera index {i} is available")
            cap.release()
    
    return available_cameras

def try_v4l2_cameras():
    """Check available video devices using v4l2."""
    try:
        print("Checking available v4l2 devices...")
        result = subprocess.run(["v4l2-ctl", "--list-devices"], 
                             capture_output=True, text=True, check=False)
        print(result.stdout)
        return True
    except Exception as e:
        print(f"v4l2-ctl check failed: {e}")
        return False

def is_raspberry_pi():
    """Check if we're running on a Raspberry Pi."""
    try:
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read()
            return 'Raspberry Pi' in model
    except:
        return False

def try_rpicamera_approach():
    """Try using the picamera2 library which works well with Raspberry Pi cameras."""
    try:
        # Check if picamera2 is installed
        import importlib.util
        if importlib.util.find_spec("picamera2") is None:
            print("picamera2 not found, try installing with 'pip install picamera2'")
            return False
            
        print("picamera2 is available, trying Raspberry Pi Camera approach...")
        
        # Create a directory to store continuous frames if needed
        if not os.path.exists("camera_frames"):
            os.makedirs("camera_frames")
        
        return True
    except Exception as e:
        print(f"picamera2 approach failed: {e}")
        return False

def capture_frame_with_rpicamera(camera_num=0):
    """Capture a single frame using picamera2."""
    try:
        from picamera2 import Picamera2
        
        # Initialize the camera
        picam2 = Picamera2(camera_num)
        
        # Configure and start the camera
        config = picam2.create_still_configuration()
        picam2.configure(config)
        picam2.start()
        
        # Wait for auto exposure to stabilize
        time.sleep(0.5)
        
        # Capture a frame
        frame = picam2.capture_array()
        
        # Convert to BGR format if needed (picamera2 uses RGB by default)
        if frame.shape[2] == 3:  # If it's a color image
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Stop the camera
        picam2.stop()
        picam2.close()
        
        return frame
    except Exception as e:
        print(f"Error capturing frame with picamera2: {e}")
        return None

def capture_frame_with_libcamera(camera_num=1):
    """Capture a single frame using libcamera (legacy approach)."""
    try:
        # Generate a unique filename based on timestamp
        timestamp = int(time.time() * 1000)
        filename = f"camera_frames/frame_{timestamp}.jpg"
        
        # Use camera_num to select the camera if supported
        cmd = ["libcamera-still", "-n", "-o", filename, "--immediate"]
        if camera_num > 0:
            cmd.extend(["--camera", str(camera_num)])
            
        # Capture image with libcamera-still
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Load the image with OpenCV
        frame = cv2.imread(filename)
        
        if frame is None or frame.size == 0:
            print("Failed to load captured image")
            return None
            
        # Optional: Remove the file to avoid filling up storage
        try:
            os.remove(filename)
        except:
            pass
        
        return frame
    except Exception as e:
        print(f"Error capturing frame with libcamera: {e}")
        return None

def fix_qt_platform_issue():
    """Fix the Qt platform plugin issue by setting the appropriate environment variable."""
    if 'QT_QPA_PLATFORM' not in os.environ:
        os.environ['QT_QPA_PLATFORM'] = 'xcb'  # Use xcb as default instead of wayland
        print("Set QT_QPA_PLATFORM=xcb to avoid Qt platform plugin issues")

def process_stereo_cameras(left_index, right_index, model, colors):
    """Process stereo cameras (left and right) for object detection and depth estimation."""
    # Setup left camera
    left_cap = cv2.VideoCapture(left_index)
    if not left_cap.isOpened():
        print(f"Failed to open left camera at index {left_index}")
        return
    
    # Setup right camera
    right_cap = cv2.VideoCapture(right_index)
    if not right_cap.isOpened():
        print(f"Failed to open right camera at index {right_index}")
        left_cap.release()
        return
    
    # Set up stereo matcher
    stereo = cv2.StereoBM.create(numDisparities=16*5, blockSize=15)
    
    # Set up windows
    cv2.namedWindow("Left Camera", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Right Camera", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Depth Map", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Left Camera", 640, 480)
    cv2.resizeWindow("Right Camera", 640, 480)
    cv2.resizeWindow("Depth Map", 640, 480)
    
    prev_time = 0
    
    # Camera parameters - ideally these would be from calibration
    # These are placeholder values - actual calibration would be needed
    focal_length = 500  # approximate focal length in pixels
    baseline = 0.1  # approximate distance between cameras in meters
    
    while True:
        # Capture frames
        ret_left, left_frame = left_cap.read()
        ret_right, right_frame = right_cap.read()
        
        if not ret_left or not ret_right:
            print("Failed to capture frames from one or both cameras")
            break
        
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
        prev_time = current_time
        
        # Convert to grayscale for stereo matching
        left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
        
        # Compute disparity map
        disparity = stereo.compute(left_gray, right_gray)
        
        # Normalize disparity for visualization
        disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
        disparity_normalized = np.uint8(disparity_normalized)
        disparity_color = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_JET)
        
        # Process left frame for detection
        left_results = model(left_frame, stream=True)
        left_result = next(left_results, None)
        
        persons_detected = []
        
        if left_result:
            left_frame = process_frame(left_frame, left_result, colors, fps)
            
            # Find persons specifically
            for i, box in enumerate(left_result.boxes):
                cls_id = int(box.cls[0])
                cls_name = left_result.names[cls_id]
                
                if cls_name.lower() == 'person':
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    # Get disparity value at the center of the person
                    if 0 <= center_y < disparity.shape[0] and 0 <= center_x < disparity.shape[1]:
                        d = disparity[center_y, center_x]
                        
                        # Calculate real-world depth (Z) using disparity
                        # Z = baseline * focal_length / disparity
                        if d > 0:  # Avoid division by zero
                            # Convert disparity to depth in meters
                            depth = baseline * focal_length / max(d, 1)
                            print(f"Person detected! Depth from stereo: {depth:.2f} meters")
                            
                            # Update audio guidance based on the detection
                            # Apply guidance if person is within 1.5 meters
                            if depth < 1.5:
                                # Update audio guidance with stereo camera detection
                                handle_multiple_people(left_frame, left_result, depth_correction_factor=1.0)
                            
                            # Draw depth information on frame
                            cv2.putText(
                                left_frame,
                                f"Stereo depth: {depth:.2f}m",
                                (x1, y2 + 25),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 0),
                                2,
                                cv2.LINE_AA
                            )
                            
                            # Mark this point on the disparity map
                            cv2.circle(disparity_color, (center_x, center_y), 5, (0, 0, 255), -1)
                            
                            persons_detected.append({
                                'position': (center_x, center_y),
                                'depth': depth
                            })
        
        # Process right frame
        right_results = model(right_frame, stream=True)
        right_result = next(right_results, None)
        if right_result:
            right_frame = process_frame(right_frame, right_result, colors, fps)
        
        # Display frames
        cv2.imshow("Left Camera", left_frame)
        cv2.imshow("Right Camera", right_frame)
        cv2.imshow("Depth Map", disparity_color)
        
        # Print overall summary of persons
        if persons_detected:
            avg_depth = sum(p['depth'] for p in persons_detected) / len(persons_detected)
            print(f"Average person depth: {avg_depth:.2f} meters ({len(persons_detected)} persons)")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    left_cap.release()
    right_cap.release()
    cv2.destroyAllWindows()

def process_frame(frame, result, colors, fps):
    """Process a single frame with detection results and provide audio guidance."""
    boxes = result.boxes
    masks = result.masks
    
    # Calibration factor based on known measurements
    # For a person at 0.762 meters, we were getting readings around 0.4 units
    # So our correction factor is 0.762/0.4 ≈ 1.905
    depth_correction_factor = 0.05
    
    # Apply segmentation masks if available
    if masks is not None:
        frame = draw_segmentation(frame, masks, boxes, result.names, colors)
    
    # Update audio guidance based on detection results
    handle_multiple_people(frame, result, depth_correction_factor)
    
    # Draw bounding boxes with labels and find persons
    for i, box in enumerate(boxes):
        # Get box coordinates and ensure they're integers
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Get class info
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        cls_name = result.names[cls_id]
        
        # Draw bounding box
        color = colors[cls_id % len(colors)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Check if the detected object is a person
        if cls_name.lower() == 'person':
            # Calculate the center of the bounding box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Calculate approximate depth (inverse of size)
            # The larger the bounding box, the closer the person
            box_width = x2 - x1
            box_height = y2 - y1
            box_size = box_width * box_height
            
            # Simple inverse relation with correction factor
            raw_depth_estimate = 1000000 / max(1, box_size)  # Raw estimate
            depth_estimate = raw_depth_estimate * depth_correction_factor
            
            # Print depth information for persons
            print(f"Person detected! Estimated depth: {depth_estimate:.2f} units")
            
            # Also display depth on the frame
            cv2.putText(
                frame,
                f"Depth: {depth_estimate:.2f}m",
                (x1, y2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
        
        # Create label with class name and confidence
        label = f"{cls_name}: {conf:.2f}"
        
        # Text settings
        font_scale = 0.7
        font_thickness = 2
        
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        
        # Draw background for text
        cv2.rectangle(
            frame, 
            (x1, y1 - text_height - baseline - 5), 
            (x1 + text_width, y1), 
            color, 
            -1
        )
        
        # Put text
        cv2.putText(
            frame, 
            label, 
            (x1, y1 - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            font_scale, 
            (255, 255, 255), 
            font_thickness, 
            cv2.LINE_AA
        )
    
    # Display FPS
    cv2.putText(
        frame, 
        f"FPS: {fps:.1f}", 
        (20, 40), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        (0, 255, 0), 
        2, 
        cv2.LINE_AA
    )
    
    return frame

def main():
    # Fix Qt platform issue
    fix_qt_platform_issue()
    
    # Load audio files first
    load_audio_files()
    
    # Load the YOLOv8 segmentation model
    print("Loading YOLOv8 model...")
    try:
        model = YOLO("yolov8n-seg.pt")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        print("Make sure the model file exists and ultralytics is installed correctly.")
        return
    
    # Define colors for visualization
    colors = generate_colors(80)
    
    # Check if we're on a Raspberry Pi first, as that's a more specific environment
    if is_raspberry_pi():
        print("Running on Raspberry Pi...")
        
        # Directly try the Raspberry Pi camera approach first
        if try_rpicamera_approach():
            print("Using picamera2 for capture...")
            process_rpicamera(model, colors)
            return
        
        # Check for available v4l2 devices which might work better on Pi
        print("Checking v4l2 devices...")
        try_v4l2_cameras()
    
    # Check for available standard cameras (might work for USB cameras on Pi too)
    print("Checking for available cameras...")
    available_cameras = check_camera_indices()
    
    if available_cameras:
        # Stereo camera mode if we have multiple cameras
        if len(available_cameras) >= 2:
            print(f"Detected multiple cameras: {available_cameras}")
            print(f"Using cameras {available_cameras[0]} and {available_cameras[1]} for stereo processing")
            process_stereo_cameras(available_cameras[0], available_cameras[1], model, colors)
            return
        elif len(available_cameras) == 1:
            print(f"Only one camera detected: {available_cameras[0]}")
            process_single_camera(available_cameras[0], model, colors)
            return
    
    # Try libcamera as a last resort on Raspberry Pi
    if is_raspberry_pi():
        print("Trying libcamera-still approach as last resort...")
        process_libcamera_fallback(model, colors)
        return
    
    print("All camera approaches failed. Exiting.")
    print("Please ensure your camera is connected properly and has appropriate permissions.")
    print("For Raspberry Pi Camera Module, ensure it's enabled in raspi-config.")
    print("For USB cameras, check permissions on /dev/video* devices.")

def process_single_camera(camera_index, model, colors):
    """Process a single camera for object detection."""
    print(f"Starting single camera processing with camera index {camera_index}")
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"Failed to open camera at index {camera_index}")
        return
    
    # Set up window
    cv2.namedWindow("YOLOv8 Object Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLOv8 Object Detection", 1280, 720)
    
    prev_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
        prev_time = current_time
        
        # Process frame
        results = model(frame, stream=True)
        result = next(results, None)
        if result:
            frame = process_frame(frame, result, colors, fps)
        
        # Display frame
        cv2.imshow("YOLOv8 Object Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

def process_rpicamera(model, colors):
    """Process frames from Raspberry Pi Camera using picamera2."""
    try:
        # Import picamera2
        from picamera2 import Picamera2
        from picamera2.outputs import FileOutput
        
        # Set up window
        cv2.namedWindow("YOLOv8 Object Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("YOLOv8 Object Detection", 1280, 720)
        
        # Initialize camera
        picam2 = Picamera2()
        
        # Configure camera for preview and capture
        preview_config = picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        picam2.configure(preview_config)
        picam2.start()
        
        prev_time = 0
        
        print("Starting camera stream with picamera2. Press 'q' to exit.")
        
        while True:
            # Capture frame
            frame = picam2.capture_array()
            
            # Convert to BGR for OpenCV processing
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
            prev_time = current_time
            
            # Process frame with YOLOv8
            try:
                results = model(frame, stream=True)
                result = next(results, None)
                if result:
                    frame = process_frame(frame, result, colors, fps)
            except Exception as e:
                print(f"Error processing frame with YOLOv8: {e}")
            
            # Display frame
            cv2.imshow("YOLOv8 Object Detection", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Clean up
        picam2.stop()
        picam2.close()
        cv2.destroyAllWindows()
        
    except ImportError:
        print("picamera2 not installed, falling back to libcamera approach")
        process_libcamera_fallback(model, colors)
    except Exception as e:
        print(f"Error with picamera2: {e}")
        print("Falling back to libcamera approach")
        process_libcamera_fallback(model, colors)

def process_libcamera_fallback(model, colors):
    """Fallback process frames using libcamera-still with person depth estimation."""
    # Set up window
    cv2.namedWindow("YOLOv8 Object Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLOv8 Object Detection", 1280, 720)
    
    prev_time = 0
    
    # Updated depth correction factor: calculated as 0.762m ÷ 3.9m = 0.195
    depth_correction_factor = 0.195
    
    print(f"Using libcamera-still fallback with depth correction factor: {depth_correction_factor}")
    print("Press 'q' to exit.")
    
    while True:
        # Capture frame using libcamera
        frame = capture_frame_with_libcamera()
        if frame is None:
            print("Failed to capture frame with libcamera-still")
            print("Retrying in 1 second...")
            time.sleep(1)
            continue
        
        # Check frame dimensions
        if frame.shape[0] <= 0 or frame.shape[1] <= 0:
            print(f"Invalid frame dimensions: {frame.shape}")
            continue
            
        # Print frame details for debugging
        print(f"Frame shape: {frame.shape}, type: {frame.dtype}")
        
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
        prev_time = current_time
        
        try:
            # Process frame
            results = model(frame, stream=True)
            result = next(results, None)
            if result:
                # Debug masks and boxes
                if hasattr(result, 'masks') and result.masks is not None:
                    print(f"Masks available: {len(result.masks)}")
                if hasattr(result, 'boxes') and result.boxes is not None:
                    print(f"Boxes available: {len(result.boxes)}")
                
                # Process frame with depth estimation
                frame = process_frame(frame, result, colors, fps)
                
                # Look specifically for persons to print their depth
                for i, box in enumerate(result.boxes):
                    cls_id = int(box.cls[0])
                    cls_name = result.names[cls_id]
                    
                    if cls_name.lower() == 'person':
                        # Calculate depth based on box size
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        box_width = x2 - x1
                        box_height = y2 - y1
                        
                        # Calculate simple depth estimation
                        # This is an approximation based on apparent size
                        # Assumes average human height of 1.7m
                        raw_depth = (1.7 * frame.shape[0]) / (box_height * 0.8)
                        
                        # Apply the correction factor
                        corrected_depth = raw_depth * depth_correction_factor
                        
                        # Display both raw and corrected depth for comparison
                        print(f"Person detected in libcamera! Raw depth: {raw_depth:.2f}m, Corrected depth: {corrected_depth:.2f}m")
                        
                        # Draw corrected depth on the frame
                        cv2.putText(
                            frame,
                            f"Depth: {corrected_depth:.2f}m",
                            (x1, y2 + 45),  # Position below any existing text
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),  # Yellow color for corrected value
                            2,
                            cv2.LINE_AA
                        )
        except Exception as e:
            print(f"Error processing frame: {e}")
        
        # Display frame
        cv2.imshow("YOLOv8 Object Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
