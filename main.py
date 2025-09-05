import cv2
import mediapipe as mp
import logging

# Set up logging to catch potential TensorFlow Lite warnings/errors
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Initialize MediaPipe Hands
try:
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=2,
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

    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )
        logger.info(f"Detected {len(results.multi_hand_landmarks)} hand(s).")

    # Convert back to BGR for display (optional, since drawing is on BGR image)
    # image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)  # Not needed here

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