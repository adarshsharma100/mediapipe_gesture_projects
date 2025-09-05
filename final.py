import cv2
import mediapipe as mp
import logging
import lgpio
import time
from gpiozero import Servo, OutputDevice
import RPi_I2C_driver

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Initialize MediaPipe Hands
try:
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils
    logger.info("MediaPipe Hands initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize MediaPipe Hands: {e}")
    exit(1)

# Initialize LCD
try:
    mylcd = RPi_I2C_driver.lcd(0x20)
    mylcd.backlight(1)
    mylcd.lcd_clear()
    logger.info("LCD initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize LCD: {e}")
    exit(1)

# Start capturing video from webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    logger.error("Failed to open webcam.")
    exit(1)

# GPIO Configuration
LED_PIN    = 12
BUZZER_PIN = 22
RELAY_PIN  = 26
SERVO_PIN  = 24
# Stepper motor pins
COIL_A_1 = 16
COIL_A_2 = 19
COIL_B_1 = 20
COIL_B_2 = 21

# Open the GPIO chip for regular GPIO pins
h = lgpio.gpiochip_open(0)

# Initialize servo
servo = Servo(SERVO_PIN, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)

# Initialize stepper motor coils
coil_A_1 = OutputDevice(COIL_A_1)
coil_A_2 = OutputDevice(COIL_A_2)
coil_B_1 = OutputDevice(COIL_B_1)
coil_B_2 = OutputDevice(COIL_B_2)

# Stepper motor speed constant
STEPPER_DELAY = 2.0 / 1000.0  # 0.002 s delay per step (high speed)

# Helper function to safely claim a GPIO as output
def safe_claim_output(pin, initial_value):
    try:
        lgpio.gpio_claim_output(h, pin)
        lgpio.gpio_write(h, pin, initial_value)
    except lgpio.error as e:
        logger.error(f"Error claiming pin {pin}: {e}")
        lgpio.gpiochip_close(h)
        exit(1)

# Initialize Actuator Pins as Outputs (set LOW)
for pin in [LED_PIN, BUZZER_PIN, RELAY_PIN]:
    safe_claim_output(pin, 0)

def set_step(w1, w2, w3, w4):
    """Set stepper motor coil states"""
    coil_A_1.value = w1
    coil_A_2.value = w2
    coil_B_1.value = w3
    coil_B_2.value = w4

def forward_stepper(delay, steps):
    """Rotate the stepper motor forward"""
    for i in range(steps):
        set_step(1, 0, 1, 0)
        time.sleep(delay)
        set_step(0, 1, 1, 0)
        time.sleep(delay)
        set_step(0, 1, 0, 1)
        time.sleep(delay)
        set_step(1, 0, 0, 1)
        time.sleep(delay)

def count_fingers(hand_landmarks):
    """Count the number of raised fingers"""
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

def control_actuators(finger_count):
    """Control actuators based on finger count and update LCD"""
    mylcd.lcd_clear()
    if finger_count == 1:
        lgpio.gpio_write(h, LED_PIN, 1)
        lgpio.gpio_write(h, BUZZER_PIN, 0)
        lgpio.gpio_write(h, RELAY_PIN, 0)
        servo.value = 0
        set_step(0, 0, 0, 0)  # Stop stepper
        mylcd.lcd_display_string("1 finger detected", 1)
        mylcd.lcd_display_string("LED is ON", 2)
        logger.info("LED ON (1 finger)")
    elif finger_count == 2:
        lgpio.gpio_write(h, LED_PIN, 0)
        lgpio.gpio_write(h, BUZZER_PIN, 1)
        time.sleep(0.5)
        lgpio.gpio_write(h, BUZZER_PIN, 0)
        lgpio.gpio_write(h, RELAY_PIN, 0)
        servo.value = 0
        set_step(0, 0, 0, 0)  # Stop stepper
        mylcd.lcd_display_string("2 fingers detected", 1)
        mylcd.lcd_display_string("Buzzer ON", 2)
        logger.info("Buzzer Beeped (2 fingers)")
    elif finger_count == 3:
        lgpio.gpio_write(h, LED_PIN, 0)
        lgpio.gpio_write(h, BUZZER_PIN, 0)
        lgpio.gpio_write(h, RELAY_PIN, 1)
        servo.value = 0
        set_step(0, 0, 0, 0)  # Stop stepper
        mylcd.lcd_display_string("3 fingers detected", 1)
        mylcd.lcd_display_string("Relay is ON", 2)
        logger.info("Relay Activated (3 fingers)")
    elif finger_count == 4:
        lgpio.gpio_write(h, LED_PIN, 0)
        lgpio.gpio_write(h, BUZZER_PIN, 0)
        lgpio.gpio_write(h, RELAY_PIN, 0)
        servo.value = 1
        set_step(0, 0, 0, 0)  # Stop stepper
        mylcd.lcd_display_string("4 fingers detected", 1)
        mylcd.lcd_display_string("Servo is ON", 2)
        logger.info("Servo Activated - Door Open (4 fingers)")
    elif finger_count == 5:
        lgpio.gpio_write(h, LED_PIN, 0)
        lgpio.gpio_write(h, BUZZER_PIN, 0)
        lgpio.gpio_write(h, RELAY_PIN, 0)
        servo.value = 0
        forward_stepper(STEPPER_DELAY, 200)  # Run stepper for 200 steps
        mylcd.lcd_display_string("5 fingers detected", 1)
        mylcd.lcd_display_string("Stepper is ON", 2)
        logger.info("Stepper Motor Activated (5 fingers)")
    else:
        lgpio.gpio_write(h, LED_PIN, 0)
        lgpio.gpio_write(h, BUZZER_PIN, 0)
        lgpio.gpio_write(h, RELAY_PIN, 0)
        servo.value = 0
        set_step(0, 0, 0, 0)  # Stop stepper
        mylcd.lcd_display_string("No fingers", 1)
        mylcd.lcd_display_string("All OFF", 2)

# Main loop
logger.info("Starting hand detection. Show 1 finger for LED, 2 for Buzzer, 3 for Relay, 4 for Servo, 5 for Stepper.")
try:
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

        # Draw hand landmarks and control actuators
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )
                
                # Count fingers and control actuators
                finger_count = count_fingers(hand_landmarks)
                control_actuators(finger_count)
                
                # Display finger count on image
                cv2.putText(
                    image,
                    f'Fingers: {finger_count}',
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )
        else:
            # No hands detected
            control_actuators(0)

        # Display the image
        cv2.imshow('MediaPipe Hands', image)
        
        # Break loop on 'q' key press
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    logger.info("Interrupted by user.")
finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    try:
        hands.close()
        lgpio.gpiochip_close(h)
        servo.value = 0
        set_step(0, 0, 0, 0)  # Stop stepper
        coil_A_1.close()
        coil_A_2.close()
        coil_B_1.close()
        coil_B_2.close()
        mylcd.lcd_clear()
        mylcd.backlight(0)
        logger.info("Resources released successfully.")
    except Exception as e:
        logger.error(f"Error releasing resources: {e}")