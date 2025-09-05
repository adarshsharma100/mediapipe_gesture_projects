import cv2
import mediapipe as mp
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize pycaw for volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Get volume range
vol_range = volume.GetVolumeRange()
min_vol = vol_range[0]
max_vol = vol_range[1]

while True:
    success, img = cap.read()
    if not success:
        break

    # Flip the image horizontally for a mirror effect
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Process the image and detect hands
    results = hands.process(img_rgb)
    
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)
            
            # Get coordinates of thumb tip (landmark 4) and index finger tip (landmark 8)
            thumb_x = hand_lms.landmark[4].x * img.shape[1]
            thumb_y = hand_lms.landmark[4].y * img.shape[0]
            index_x = hand_lms.landmark[8].x * img.shape[1]
            index_y = hand_lms.landmark[8].y * img.shape[0]
            
            # Calculate the distance between thumb and index finger
            distance = math.hypot(index_x - thumb_x, index_y - thumb_y)
            
            # Convert the distance to volume level (map 50-300 pixels to volume range)
            vol = np.interp(distance, [50, 300], [min_vol, max_vol])
            vol_bar = np.interp(distance, [50, 300], [400, 150])
            vol_per = np.interp(distance, [50, 300], [0, 100])
            
            # Set system volume
            volume.SetMasterVolumeLevel(vol, None)
            
            # Draw volume bar
            cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
            cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, f'{int(vol_per)} %', (40, 450), cv2.FONT_HERSHEY_SIMPLEX,
                       1, (0, 255, 0), 2)
            
            # Draw line between thumb and index finger
            cv2.line(img, (int(thumb_x), int(thumb_y)), 
                    (int(index_x), int(index_y)), (255, 0, 0), 3)
            cv2.circle(img, (int(thumb_x), int(thumb_y)), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (int(index_x), int(index_y)), 10, (0, 0, 255), cv2.FILLED)

    # Display the image
    cv2.imshow("Hand Gesture Volume Control", img)
    
    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()