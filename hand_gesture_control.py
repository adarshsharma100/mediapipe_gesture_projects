import cv2
import mediapipe as mp
import numpy as np
from math import sqrt, hypot
import lgpio
import time
from pulsectl import Pulse
from gpiozero import Servo

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# Initialize video capture
cap = cv2.VideoCapture(0)

# GPIO setup
LED_PIN = 12
BUZZER_PIN = 22
RELAY_PIN = 26
SERVO_PIN = 24

# Open the GPIO chip
try:
    h = lgpio.gpiochip_open(0)
    if h < 0:
        raise RuntimeError("Failed to open GPIO chip")
except Exception as e:
    print(f"GPIO chip open failed: {e}")
    exit(1)

# Initialize PulseAudio control
pulse = Pulse('hand-gesture-control')

# Setup servo
servo = Servo(SERVO_PIN, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)

# Helper function to safely claim a GPIO as output
def safe_claim_output(handle, pin, initial_value):
    try:
        lgpio.gpio_claim_output(handle, pin)
        lgpio.gpio_write(handle, pin, initial_value)
    except lgpio.error as e:
        print(f"Error claiming pin {pin}: {e}")
        lgpio.gpiochip_close(handle)
        exit(1)

# Initialize GPIO pins
safe_claim_output(h, LED_PIN, 0)
safe_claim_output(h, BUZZER_PIN, 0)
safe_claim_output(h, RELAY_PIN, 0)

# Get frame width dynamically after capturing first frame
ret, frame = cap.read()
if ret:
    frame_width = frame.shape[1]
else:
    frame_width = 640  # Default width if frame capture fails

# Button properties - positioned vertically on the rightmost side
BUTTON_WIDTH = 100
BUTTON_HEIGHT = 50
RIGHT_MARGIN = 20  # Margin from right edge
buttons = {
    'LED': {'x': frame_width - BUTTON_WIDTH - RIGHT_MARGIN, 'y': 100, 'w': BUTTON_WIDTH, 'h': BUTTON_HEIGHT, 'state': False},
    'Buzzer': {'x': frame_width - BUTTON_WIDTH - RIGHT_MARGIN, 'y': 180, 'w': BUTTON_WIDTH, 'h': BUTTON_HEIGHT, 'state': False},
    'Relay': {'x': frame_width - BUTTON_WIDTH - RIGHT_MARGIN, 'y': 260, 'w': BUTTON_WIDTH, 'h': BUTTON_HEIGHT, 'state': False}
}

def calculate_distance(p1, p2):
    """Calculate distance between two points"""
    return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Volume box coordinates (made larger: wider and taller)
VOL_BOX_X1, VOL_BOX_Y1 = 40, 120    # Moved left and up slightly
VOL_BOX_X2, VOL_BOX_Y2 = 110, 450   # Increased width (from 85 to 110) and height (from 400 to 450)

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    # Draw buttons
    for button_name, props in buttons.items():
        button_color = (0, 255, 0) if props['state'] else (0, 0, 255)
        cv2.rectangle(frame, (props['x'], props['y']), 
                     (props['x'] + props['w'], props['y'] + props['h']), 
                     button_color, -1)
        status_text = "ON" if props['state'] else "OFF"
        text_x = props['x'] + 10 if button_name != 'Relay' else props['x'] + 5
        cv2.putText(frame, button_name, 
                   (text_x, props['y'] + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, status_text, 
                   (text_x, props['y'] + 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Draw volume box (always visible)
    cv2.rectangle(frame, (VOL_BOX_X1, VOL_BOX_Y1), (VOL_BOX_X2, VOL_BOX_Y2), (0, 255, 0), 3)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            index_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]
            
            h_frame, w, _ = frame.shape
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h_frame)
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h_frame)
            
            # Distance for button control
            button_distance = calculate_distance((index_x, index_y), (thumb_x, thumb_y))
            
            # Distance for volume/servo control
            vol_servo_distance = hypot(index_x - thumb_x, index_y - thumb_y)
            
            # Button control logic
            for button_name, props in buttons.items():
                finger_over_button = (props['x'] <= index_x <= props['x'] + props['w'] and 
                                    props['y'] <= index_y <= props['y'] + props['h'])
                
                if finger_over_button and button_distance < 30:
                    cv2.waitKey(200)  # Debounce delay
                    buttons[button_name]['state'] = not buttons[button_name]['state']
                    
                    if button_name == 'LED':
                        lgpio.gpio_write(h, LED_PIN, 1 if buttons['LED']['state'] else 0)
                        print(f"LED {'ON' if buttons['LED']['state'] else 'OFF'}")
                    elif button_name == 'Buzzer':
                        lgpio.gpio_write(h, BUZZER_PIN, 1 if buttons['Buzzer']['state'] else 0)
                        print(f"Buzzer {'ON' if buttons['Buzzer']['state'] else 'OFF'}")
                    elif button_name == 'Relay':
                        lgpio.gpio_write(h, RELAY_PIN, 1 if buttons['Relay']['state'] else 0)
                        print(f"Relay {'Activated' if buttons['Relay']['state'] else 'Deactivated'}")
            
            # Check if hand is in volume box
            hand_in_vol_box = (VOL_BOX_X1 <= index_x <= VOL_BOX_X2 and 
                             VOL_BOX_Y1 <= index_y <= VOL_BOX_Y2 and
                             VOL_BOX_X1 <= thumb_x <= VOL_BOX_X2 and 
                             VOL_BOX_Y1 <= thumb_y <= VOL_BOX_Y2)
            
            if hand_in_vol_box:
                # Volume control
                vol = np.interp(vol_servo_distance, [50, 300], [0, 1])
                vol_bar = np.interp(vol_servo_distance, [50, 300], [450, 120])  # Adjusted for new height
                vol_per = np.interp(vol_servo_distance, [50, 300], [0, 100])
                sink = pulse.sink_list()[0]
                pulse.volume_set_all_chans(sink, vol)
                
                # Servo control
                servo_val = np.interp(vol_servo_distance, [50, 300], [-1, 1])
                servo.value = servo_val
                
                # Draw volume bar and text
                cv2.rectangle(frame, (40, int(vol_bar)), (110, 450), (0, 255, 0), cv2.FILLED)  # Adjusted for new size
                cv2.putText(frame, f'{int(vol_per)} %', (30, 480), cv2.FONT_HERSHEY_SIMPLEX,
                           1, (0, 255, 0), 2)
                cv2.putText(frame, f'Servo: {int((servo_val + 1) * 90)} deg', (30, 510),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # Draw hand indicators (always visible)
            cv2.circle(frame, (index_x, index_y), 10, (255, 0, 0), -1)
            cv2.circle(frame, (thumb_x, thumb_y), 10, (0, 0, 255), cv2.FILLED)
            cv2.line(frame, (index_x, index_y), (thumb_x, thumb_y), (255, 0, 0), 3)
    
    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('1'):
        buttons['LED']['state'] = True
        lgpio.gpio_write(h, LED_PIN, 1)
        print("LED ON")
    elif key == ord('2'):
        buttons['LED']['state'] = False
        lgpio.gpio_write(h, LED_PIN, 0)
        print("LED OFF")
    elif key == ord('3'):
        lgpio.gpio_write(h, BUZZER_PIN, 1)
        time.sleep(0.5)
        lgpio.gpio_write(h, BUZZER_PIN, 0)
        buttons['Buzzer']['state'] = False
        print("Buzzer Beeped")
    elif key == ord('4'):
        buttons['Relay']['state'] = True
        lgpio.gpio_write(h, RELAY_PIN, 1)
        print("Relay Activated")
    elif key == ord('5'):
        buttons['Relay']['state'] = False
        lgpio.gpio_write(h, RELAY_PIN, 0)
        print("Relay Deactivated")
    elif key == ord('q'):
        break

    cv2.imshow('Hand Gesture Control Panel', frame)

# Cleanup
cap.release()
cv2.destroyAllWindows()
hands.close()
lgpio.gpiochip_close(h)
servo.value = 0
pulse.close()