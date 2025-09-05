import cv2
import mediapipe as mp
import numpy as np
import math

class GestureShapeRecognizer:
    def __init__(self):
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize variables for tracking finger positions
        self.points = []
        self.is_tracking = False
        self.last_shape = None
        self.shape_confirmed = False
        
        # Constants for shape detection
        self.MIN_POINTS = 15  # Minimum points needed for shape recognition
        self.TIMEOUT = 30  # Frames to wait after shape detection before recognizing again

    def recognize_shape(self, points):
        """Recognize shape based on collected points"""
        if len(points) < self.MIN_POINTS:
            return None
            
        # Convert points to numpy array
        points = np.array(points, dtype=np.int32)
        
        # Calculate perimeter and area
        perimeter = cv2.arcLength(points, True)
        area = cv2.contourArea(points)
        
        # Skip if the area is too small (noise)
        if area < 1000:
            return None
            
        # Calculate circularity: 4 * pi * area / perimeter^2
        # A perfect circle has circularity of 1
        circularity = 4 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Get the bounding rectangle
        x, y, w, h = cv2.boundingRect(points)
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # Fit the points to a rotated rectangle
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        rect_area = rect[1][0] * rect[1][1]
        
        # Calculate fullness (area of contour / area of minimum rotated rectangle)
        fullness = area / rect_area if rect_area > 0 else 0
        
        # Calculate convex hull and convexity
        hull = cv2.convexHull(points)
        hull_area = cv2.contourArea(hull)
        convexity = area / hull_area if hull_area > 0 else 0
        
        # Decision logic for shapes
        if circularity > 0.7:
            return "Circle"
        elif 0.8 < aspect_ratio < 1.2 and convexity > 0.8 and fullness > 0.75:
            return "Square"
        
        return None

    def process_frame(self, frame):
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe
        results = self.hands.process(rgb_frame)
        
        # Draw hand landmarks on the frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Get the index fingertip position
                index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                h, w, _ = frame.shape
                x, y = int(index_tip.x * w), int(index_tip.y * h)
                
                # Draw a circle at the fingertip position
                cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
                
                # Track the fingertip when it's raised
                if index_tip.y < hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP].y:
                    if not self.is_tracking:
                        self.is_tracking = True
                        self.points = []
                        self.shape_confirmed = False
                    
                    # Add the point to the tracking list
                    self.points.append(np.array([x, y]))
                else:
                    # When finger is lowered, try to recognize the shape
                    if self.is_tracking and len(self.points) >= self.MIN_POINTS:
                        self.is_tracking = False
                        shape = self.recognize_shape(np.array(self.points))
                        if shape:
                            self.last_shape = shape
                            self.shape_confirmed = True
                            print(f"Detected: {shape}")
                            # Reset after recognition
                            self.points = []
                    elif self.is_tracking:
                        self.is_tracking = False
                        self.points = []
        
        # Draw the tracked path
        if len(self.points) > 1:
            points_array = np.array(self.points, dtype=np.int32)
            cv2.polylines(frame, [points_array], False, (0, 0, 255), 2)
        
        # Display the detected shape
        if self.shape_confirmed:
            cv2.putText(frame, f"Shape: {self.last_shape}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Draw the recognized shape
            if self.last_shape == "Circle":
                center = (320, 240)  # Center of the frame
                cv2.circle(frame, center, 100, (255, 0, 0), 3)
            elif self.last_shape == "Square":
                top_left = (220, 140)
                bottom_right = (420, 340)
                cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 3)
                
            # Decrease the timeout counter
            self.timeout_counter = self.TIMEOUT
            
        return frame

def main():
    cap = cv2.VideoCapture(0)
    recognizer = GestureShapeRecognizer()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process the frame
        processed_frame = recognizer.process_frame(frame)
        
        # Display the frame
        cv2.imshow('MediaPipe Gesture Shape Recognition', processed_frame)
        
        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()