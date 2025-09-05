import cv2
import mediapipe as mp
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Initialize MediaPipe Hands
try:
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,  # Changed to 1 hand for simplicity
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils
    logger.info("MediaPipe Hands initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize MediaPipe Hands: {e}")
    exit(1)

# Start capturing video from webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    logger.error("Failed to open webcam.")
    exit(1)

def count_fingers(hand_landmarks):
    """Count the number of raised fingers"""
    # Landmark indices for fingertips
    finger_tips = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
    count = 0
    
    # Get wrist y-coordinate as reference
    wrist_y = hand_landmarks.landmark[0].y
    
    # Count thumb (special case - checking if it's to the right/left of index base)
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[5].x:  # Right hand
        if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
            count += 1
    else:  # Left hand
        if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x:
            count += 1
    
    # Count other fingers (check if tip is above pip joint)
    for tip_idx in finger_tips[1:]:  # Skip thumb
        tip_y = hand_landmarks.landmark[tip_idx].y
        pip_y = hand_landmarks.landmark[tip_idx - 2].y  # PIP joint
        if tip_y < pip_y:  # Finger is raised if tip is above PIP
            count += 1
            
    return count

while cap.isOpened():
    success, image = cap.read()
    if not success:
        logger.warning("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally and convert to RGB
    image = cv2.flip(image, 1)
    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except cv2.error as e:
        logger.error(f"Failed to convert image to RGB: {e}")
        continue

    # Process the image and detect hands
    try:
        results = hands.process(image_rgb)
    except Exception as e:
        logger.error(f"Error processing image with MediaPipe: {e}")
        continue

    # Draw hand landmarks and count fingers
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )
            
            # Count fingers
            finger_count = count_fingers(hand_landmarks)
            
            # Display finger count on image
            cv2.putText(
                image,
                f'Fingers: {finger_count}',
                (10, 30),  # Position
                cv2.FONT_HERSHEY_SIMPLEX,
                1,  # Font scale
                (0, 255, 0),  # Green color
                2,  # Thickness
                cv2.LINE_AA
            )
            
        logger.info(f"Detected {len(results.multi_hand_landmarks)} hand(s) with {finger_count} finger(s).")

    # Display the image
    cv2.imshow('MediaPipe Hands', image)
    
    # Break loop on 'q' key press
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
try:
    hands.close()
    logger.info("Resources released successfully.")
except Exception as e:
    logger.error(f"Error closing MediaPipe Hands: {e}")