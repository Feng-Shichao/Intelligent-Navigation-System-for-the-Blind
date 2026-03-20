import os
import time
import numpy as np
import pygame
import cv2
import threading

# Global variables
sound_files = {}
last_audio_time = 0
audio_cooldown = 3  # seconds between audio cues
audio_thread = None
audio_queue = []
is_audio_initialized = False
audio_lock = threading.Lock()

# Distance thresholds for audio guidance (in estimated depth units)
MIN_DETECTION_RANGE = 0.5  # Minimum distance to start detecting obstacles
MAX_DETECTION_RANGE = 3.0  # Maximum distance to provide audio guidance
STOP_THRESHOLD = 0.5  # Play "stop" when obstacle is within this distance

def load_audio_files():
    """Load all audio files for feedback."""
    global sound_files, is_audio_initialized
    
    try:
        # Initialize pygame mixer
        pygame.mixer.init()
        is_audio_initialized = True
        
        # Directory containing audio files
        audio_dir = "audio"
        
        # Create directory if it doesn't exist
        if not os.path.exists(audio_dir):
            os.makedirs(audio_dir)
            print(f"Created audio directory at {os.path.abspath(audio_dir)}")
            print("Please place audio files in this directory")
        
        # Define file paths for all audio cues
        sound_paths = {
            "person_detected": os.path.join(audio_dir, "person_detected.mp3"),
            "obstacle_ahead": os.path.join(audio_dir, "obstacle_ahead.mp3"),
            "turn_left": os.path.join(audio_dir, "move_left.mp3"),  # Using move_left.mp3 as requested
            "turn_right": os.path.join(audio_dir, "move_right.mp3"),  # Using move_right.mp3 as requested
            "clear_path": os.path.join(audio_dir, "clear_path.mp3"),
            "stop": os.path.join(audio_dir, "stop.mp3"),
        }
        
        # Check which audio files exist and load them
        for name, path in sound_paths.items():
            if os.path.exists(path):
                sound_files[name] = path
                print(f"Loaded audio file: {name}")
            else:
                print(f"Warning: Audio file not found: {path}")
                # Create a placeholder text file to help user know what files are needed
                with open(f"{path}.txt", "w") as f:
                    f.write(f"Please place {os.path.basename(path)} here for {name} audio cue")
        
        # Start audio playback thread
        start_audio_thread()
        
        print(f"Audio initialization complete. Found {len(sound_files)} audio files.")
    except Exception as e:
        print(f"Error initializing audio: {e}")
        is_audio_initialized = False

def play_audio(sound_name):
    """Add an audio cue to the playback queue."""
    global audio_queue, audio_lock
    
    if not is_audio_initialized:
        print(f"Audio not initialized. Would play: {sound_name}")
        return
    
    if sound_name not in sound_files:
        print(f"Warning: Sound '{sound_name}' not found in loaded sounds")
        return
    
    # Add sound to queue with thread safety
    with audio_lock:
        audio_queue.append(sound_name)
        print(f"Queued audio: {sound_name}")

def audio_playback_thread():
    """Thread function to play audio files from queue."""
    global audio_queue, audio_lock
    
    print("Audio playback thread started")
    
    while True:
        sound_to_play = None
        
        # Check if there's a sound to play
        with audio_lock:
            if audio_queue:
                sound_to_play = audio_queue.pop(0)
        
        # Play sound if one was found
        if sound_to_play and sound_to_play in sound_files:
            try:
                print(f"Playing audio: {sound_to_play}")
                pygame.mixer.music.load(sound_files[sound_to_play])
                pygame.mixer.music.play()
                
                # Wait for the audio to finish
                while pygame.mixer.music.get_busy():
                    pygame.time.delay(100)
            except Exception as e:
                print(f"Error playing audio {sound_to_play}: {e}")
        
        # Small sleep to prevent high CPU usage
        time.sleep(0.1)

def start_audio_thread():
    """Start the audio playback thread."""
    global audio_thread
    
    if audio_thread is None or not audio_thread.is_alive():
        audio_thread = threading.Thread(target=audio_playback_thread, daemon=True)
        audio_thread.start()
        print("Started audio playback thread")

def analyze_space_around_person(frame, person_box, masks=None):
    """
    Analyze available space around a detected person to determine best direction.
    
    Args:
        frame: Camera frame
        person_box: Bounding box of detected person [x1, y1, x2, y2]
        masks: Segmentation masks if available
        
    Returns:
        direction: "turn_left", "turn_right", or "stop"
    """
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = person_box
    
    # Calculate person position relative to frame center
    person_center_x = (x1 + x2) // 2
    frame_center_x = frame_width // 2
    
    # Check if person is centered or to one side
    person_position = "center"
    center_threshold = frame_width * 0.2  # 20% of width
    
    if person_center_x < frame_center_x - center_threshold:
        person_position = "left"
    elif person_center_x > frame_center_x + center_threshold:
        person_position = "right"
    
    # Divide frame into left and right sections for obstacle analysis
    left_section = frame[:, :frame_width//2]
    right_section = frame[:, frame_width//2:]
    
    # Use segmentation masks if available for better obstacle detection
    left_obstacle_score = 0
    right_obstacle_score = 0
    
    if masks is not None and len(masks) > 0:
        # Create obstacle mask
        obstacle_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        
        for mask in masks:
            if hasattr(mask, 'data'):
                mask_np = mask.data.cpu().numpy().astype(np.uint8)
                mask_resized = cv2.resize(mask_np, (frame_width, frame_height))
                obstacle_mask = np.logical_or(obstacle_mask, mask_resized)
        
        # Count obstacle pixels in left and right sections
        left_obstacle_count = np.sum(obstacle_mask[:, :frame_width//2])
        right_obstacle_count = np.sum(obstacle_mask[:, frame_width//2:])
        
        # Normalize by section size
        left_obstacle_score = left_obstacle_count / (frame_height * frame_width//2)
        right_obstacle_score = right_obstacle_count / (frame_height * frame_width//2)
    else:
        # Fallback: Simple edge detection for obstacles
        left_gray = cv2.cvtColor(left_section, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_section, cv2.COLOR_BGR2GRAY)
        
        left_edges = cv2.Canny(left_gray, 50, 150)
        right_edges = cv2.Canny(right_gray, 50, 150)
        
        left_obstacle_score = np.sum(left_edges) / (frame_height * frame_width//2)
        right_obstacle_score = np.sum(right_edges) / (frame_height * frame_width//2)
    
    # Determine best direction based on person position and obstacle scores
    if left_obstacle_score < right_obstacle_score * 0.7:
        return "turn_left"  # Left is clearer
    elif right_obstacle_score < left_obstacle_score * 0.7:
        return "turn_right"  # Right is clearer
    elif person_position == "left":
        return "turn_right"  # Person is on left, suggest moving right
    elif person_position == "right":
        return "turn_left"  # Person is on right, suggest moving left
    else:
        # Person is centered and obstacles are similar on both sides
        if left_obstacle_score > 0.1 and right_obstacle_score > 0.1:
            return "stop"  # Too many obstacles both ways
        return "turn_left" if left_obstacle_score <= right_obstacle_score else "turn_right"

def update_audio_guidance(frame, detection_result, depth_correction_factor=1.0):
    """
    Update audio guidance based on detection results.
    
    Args:
        frame: Camera frame
        detection_result: YOLO detection results
        depth_correction_factor: Factor to adjust depth estimates
    """
    global last_audio_time
    
    # Check if enough time has elapsed since last audio
    current_time = time.time()
    if current_time - last_audio_time < audio_cooldown:
        return
    
    # Process person detections and find the closest one
    boxes = detection_result.boxes
    masks = detection_result.masks if hasattr(detection_result, 'masks') else None
    
    closest_person = None
    min_depth = float('inf')
    person_box = None
    
    for i, box in enumerate(boxes):
        # Get class info
        cls_id = int(box.cls[0])
        cls_name = detection_result.names[cls_id]
        
        # Check if the detected object is a person
        if cls_name.lower() == 'person':
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Calculate approximate depth based on box size
            box_width = x2 - x1
            box_height = y2 - y1
            box_size = box_width * box_height
            
            # Apply inverse relation with correction factor
            depth_estimate = (1000000 / max(1, box_size)) * depth_correction_factor
            
            # Track closest person
            if depth_estimate < min_depth:
                min_depth = depth_estimate
                closest_person = cls_name
                person_box = [x1, y1, x2, y2]
    
    # Check if obstacle is very close (under STOP_THRESHOLD)
    if closest_person and min_depth < STOP_THRESHOLD:
        print(f"Person detected at {min_depth:.2f} meters - TOO CLOSE, STOPPING")
        play_audio("stop")
        last_audio_time = current_time
        return
    
    # If a person is detected within threshold distance range
    if closest_person and MIN_DETECTION_RANGE <= min_depth <= MAX_DETECTION_RANGE:
        print(f"Person detected at {min_depth:.2f} meters - within guidance range")
        
        # First, notify about person detection
        play_audio("person_detected")
        
        # Analyze available space and determine direction
        if person_box:
            direction = analyze_space_around_person(frame, person_box, masks)
            print(f"Suggested direction: {direction}")
            
            # Play appropriate directional audio
            play_audio(direction)
        
        # Update last audio time
        last_audio_time = current_time
    elif closest_person:
        # Person detected but outside guidance range, just log this
        if min_depth < MIN_DETECTION_RANGE:
            print(f"Person too close ({min_depth:.2f}m) - no audio guidance needed")
        else:
            print(f"Person too far ({min_depth:.2f}m) - out of guidance range")

def handle_multiple_people(frame, detection_result, depth_correction_factor=1.0):
    """
    Handle navigation around multiple people, finding open spaces to guide user.
    
    This function analyzes the positions and depths of multiple people in the frame,
    identifies the best path to navigate through, and provides audio guidance.
    
    Args:
        frame: Camera frame
        detection_result: YOLO detection results
        depth_correction_factor: Factor to adjust depth estimates
    """
    global last_audio_time
    
    # Check if enough time has elapsed since last audio
    current_time = time.time()
    if current_time - last_audio_time < audio_cooldown:
        return
    
    # Process all person detections
    boxes = detection_result.boxes
    masks = detection_result.masks if hasattr(detection_result, 'masks') else None
    frame_height, frame_width = frame.shape[:2]
    frame_center_x = frame_width // 2
    
    # Collect all person detections
    person_detections = []
    
    for i, box in enumerate(boxes):
        # Get class info
        cls_id = int(box.cls[0])
        cls_name = detection_result.names[cls_id]
        
        # Check if the detected object is a person
        if cls_name.lower() == 'person':
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Calculate approximate depth based on box size
            box_width = x2 - x1
            box_height = y2 - y1
            box_size = box_width * box_height
            
            # Apply inverse relation with correction factor
            depth_estimate = (1000000 / max(1, box_size)) * depth_correction_factor
            
            # Calculate horizontal position relative to center
            person_center_x = (x1 + x2) // 2
            position = "center"
            if person_center_x < frame_center_x:
                position = "left"
            else:
                position = "right"
            
            # Store detection information
            person_detections.append({
                'box': [x1, y1, x2, y2],
                'depth': depth_estimate,
                'position': position,
                'center_x': person_center_x
            })
    
    # If no people detected, no guidance needed
    if not person_detections:
        return
    
    # Check if any person is too close (under STOP_THRESHOLD)
    closest_person = min(person_detections, key=lambda p: p['depth'])
    if closest_person['depth'] < STOP_THRESHOLD:
        print(f"Person detected at {closest_person['depth']:.2f} meters - TOO CLOSE, STOPPING")
        play_audio("stop")
        last_audio_time = current_time
        return
    
    # Check if any people are within detection range
    in_range_detections = [p for p in person_detections if MIN_DETECTION_RANGE <= p['depth'] <= MAX_DETECTION_RANGE]
    
    # If no people within range, skip guidance
    if not in_range_detections:
        closest_depth = min([p['depth'] for p in person_detections])
        if closest_depth < MIN_DETECTION_RANGE:
            print(f"People detected but too close ({closest_depth:.2f}m) - out of guidance range")
        else:
            print(f"People detected but too far ({closest_depth:.2f}m) - out of guidance range")
        return
    
    # Proceed with guidance for in-range detections
    print(f"People detected within guidance range: {len(in_range_detections)}")
    
    # Count people on left and right sides
    left_people = [p for p in in_range_detections if p['position'] == 'left']
    right_people = [p for p in in_range_detections if p['position'] == 'right']
    
    # Calculate average depths on each side
    left_avg_depth = sum(p['depth'] for p in left_people) / len(left_people) if left_people else float('inf')
    right_avg_depth = sum(p['depth'] for p in right_people) / len(right_people) if right_people else float('inf')
    
    # Calculate open space scores
    # Higher score = more open space (fewer people, greater average distance)
    left_openness = left_avg_depth * (1.0 / max(1, len(left_people)))
    right_openness = right_avg_depth * (1.0 / max(1, len(right_people)))
    
    # Debug information
    print(f"Left people: {len(left_people)}, avg depth: {left_avg_depth:.2f}m, openness: {left_openness:.2f}")
    print(f"Right people: {len(right_people)}, avg depth: {right_avg_depth:.2f}m, openness: {right_openness:.2f}")
    
    # Determine best direction based on open space analysis
    # Consider immediate obstacles (people closer than 1.5m but still within range)
    close_left = any(MIN_DETECTION_RANGE <= p['depth'] < 1.5 for p in left_people)
    close_right = any(MIN_DETECTION_RANGE <= p['depth'] < 1.5 for p in right_people)
    
    # Visualization: draw left/right zones and openness scores
    cv2.line(frame, (frame_center_x, 0), (frame_center_x, frame_height), (255, 255, 255), 1)
    cv2.putText(frame, f"L: {left_openness:.1f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"R: {right_openness:.1f}", (frame_width - 120, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Determine guidance direction
    direction = None
    
    # If one side has close obstacles and the other doesn't, prefer the clear side
    if close_left and not close_right:
        direction = "turn_right"
    elif close_right and not close_left:
        direction = "turn_left"
    # Otherwise, go with the more open side (higher openness score)
    elif left_openness > right_openness * 1.2:  # 20% more open to trigger a direction change
        direction = "turn_left"
    elif right_openness > left_openness * 1.2:
        direction = "turn_right"
    # If both sides are similarly occupied, suggest stopping
    elif close_left and close_right:
        direction = "stop"
    # If no clear direction but at least one person detected in range
    elif len(in_range_detections) > 0:
        # Find closest person for standard guidance
        closest_person = min(in_range_detections, key=lambda p: p['depth'])
        # Use standard space analysis for the closest person
        direction = analyze_space_around_person(frame, closest_person['box'], masks)
    
    # Provide audio guidance if we have a direction
    if direction:
        print(f"Multiple people guidance: {direction}")
        play_audio(direction)
        
        # Mark the chosen direction on frame
        if direction == "turn_left":
            arrow_start = (frame_center_x, frame_height // 2)
            arrow_end = (frame_width // 4, frame_height // 2)
            cv2.arrowedLine(frame, arrow_start, arrow_end, (0, 255, 0), 3)
        elif direction == "turn_right":
            arrow_start = (frame_center_x, frame_height // 2)
            arrow_end = (3 * frame_width // 4, frame_height // 2)
            cv2.arrowedLine(frame, arrow_start, arrow_end, (0, 255, 0), 3)
        elif direction == "stop":
            cv2.putText(frame, "STOP", (frame_center_x - 40, frame_height // 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        # Update last audio time
        last_audio_time = current_time