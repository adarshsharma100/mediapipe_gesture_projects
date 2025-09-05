import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize variables
cap = cv2.VideoCapture(0)  # 0 for webcam
person_counter = 0
tracked_faces = {}  # Dictionary to store face ID and position
left_to_right_count = set()
right_to_left_count = set()
last_positions = {}
frame_count = 0  # To periodically clean up old tracks

# Define single vertical counting line
LINE_POSITION = 320
LINE_COLOR = (0, 255, 0)    # Green
COUNT_COLOR = (0, 0, 255)   # Red
SQUARE_COLOR = (255, 0, 0)  # Blue

def calculate_face_center(detection, width, height):
    """Calculate center of face bounding box"""
    bbox = detection.location_data.relative_bounding_box
    x = int((bbox.xmin + bbox.width / 2) * width)
    y = int((bbox.ymin + bbox.height / 2) * height)
    return (x, y)

def get_face_size(detection, width, height):
    """Calculate size of face for better tracking"""
    bbox = detection.location_data.relative_bounding_box
    return int(bbox.width * width), int(bbox.height * height)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = frame.shape[:2]
    
    # Process frame
    results = face_detection.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Draw counting line
    cv2.line(image, (LINE_POSITION, 0), (LINE_POSITION, height), LINE_COLOR, 2)
    
    current_positions = {}
    current_detections = []
    
    if results.detections:
        # Store all current detections
        for detection in results.detections:
            center = calculate_face_center(detection, width, height)
            current_detections.append((center, detection))
        
        # Match with existing tracked faces or assign new IDs
        unmatched_detections = current_detections.copy()
        
        # Update existing tracks
        for person_id in list(tracked_faces.keys()):
            if not unmatched_detections:
                break
                
            last_pos = tracked_faces[person_id]
            min_dist = float('inf')
            best_match_idx = -1
            
            # Find closest detection
            for i, (center, detection) in enumerate(unmatched_detections):
                dist = np.sqrt((center[0] - last_pos[0])**2 + 
                             (center[1] - last_pos[1])**2)
                if dist < min_dist and dist < 150:  # Increased threshold
                    min_dist = dist
                    best_match_idx = i
            
            if best_match_idx != -1:
                center, detection = unmatched_detections.pop(best_match_idx)
                tracked_faces[person_id] = center
                current_positions[person_id] = center
                
                # Draw detection
                bbox = detection.location_data.relative_bounding_box
                x_min = int(bbox.xmin * width)
                y_min = int(bbox.ymin * height)
                box_width, box_height = get_face_size(detection, width, height)
                box_size = max(box_width, box_height)
                x_min = int(center[0] - box_size / 2)
                y_min = int(center[1] - box_size / 2)
                x_max = min(width, x_min + box_size)
                y_max = min(height, y_min + box_size)
                
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), 
                            SQUARE_COLOR, 2)
                cv2.putText(image, f"ID: {person_id}", 
                          (x_min, y_min - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, SQUARE_COLOR, 2)
        
        # Add new faces
        for center, detection in unmatched_detections:
            person_id = person_counter
            person_counter += 1
            tracked_faces[person_id] = center
            current_positions[person_id] = center
            
            # Draw new detection
            bbox = detection.location_data.relative_bounding_box
            x_min = int(bbox.xmin * width)
            y_min = int(bbox.ymin * height)
            box_width, box_height = get_face_size(detection, width, height)
            box_size = max(box_width, box_height)
            x_min = int(center[0] - box_size / 2)
            y_min = int(center[1] - box_size / 2)
            x_max = min(width, x_min + box_size)
            y_max = min(height, y_min + box_size)
            
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), 
                         SQUARE_COLOR, 2)
            cv2.putText(image, f"ID: {person_id}", 
                       (x_min, y_min - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, SQUARE_COLOR, 2)
    
    # Check crossings and update counts
    for person_id, current_pos in current_positions.items():
        if person_id in last_positions:
            last_x = last_positions[person_id][0]
            current_x = current_pos[0]
            
            if (person_id not in left_to_right_count and 
                last_x < LINE_POSITION and 
                current_x > LINE_POSITION):
                left_to_right_count.add(person_id)
                print(f"Face {person_id} crossed Left to Right!")
                
            if (person_id not in right_to_left_count and 
                last_x > LINE_POSITION and 
                current_x < LINE_POSITION):
                right_to_left_count.add(person_id)
                print(f"Face {person_id} crossed Right to Left!")
    
    # Update last positions
    last_positions = current_positions.copy()
    
    # Periodic cleanup of old tracks (every 30 frames)
    if frame_count % 30 == 0:
        tracked_faces = {pid: pos for pid, pos in current_positions.items()}
    
    # Display counts
    ltr_count = len(left_to_right_count)
    rtl_count = len(right_to_left_count)
    total_faces = len(tracked_faces)
    cv2.putText(image, f"L->R: {ltr_count}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, COUNT_COLOR, 2)
    cv2.putText(image, f"R->L: {rtl_count}", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, COUNT_COLOR, 2)
    cv2.putText(image, f"Total Faces: {total_faces}", 
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, COUNT_COLOR, 2)
    
    cv2.imshow('MediaPipe Face Tracking', image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
face_detection.close()