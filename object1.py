import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# Categories from COCO dataset
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Configure the Object Detector
    base_options = python.BaseOptions(model_asset_path='efficientdet_lite0.tflite')
    options = vision.ObjectDetectorOptions(
        base_options=base_options,
        score_threshold=0.5,
        max_results=10)
    
    detector = vision.ObjectDetector.create_from_options(options)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect objects
        detection_result = detector.detect(mp_image)

        # Process detections
        detected_objects = set()
        
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            category = detection.categories[0]
            
            # Handle case where index might be None or invalid
            class_index = category.index if category.index is not None else 0
            class_index = min(max(class_index, 0), len(COCO_CLASSES) - 1)
            class_name = COCO_CLASSES[class_index]
            confidence = category.score

            # Draw bounding box
            cv2.rectangle(frame, 
                         (bbox.origin_x, bbox.origin_y),
                         (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height),
                         (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(frame, label, 
                       (bbox.origin_x, bbox.origin_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            detected_objects.add(class_name)

        # Print detected objects
        if detected_objects:
            print("Detected objects:", ", ".join(detected_objects))

        # Show frame
        cv2.imshow('MediaPipe Object Detection', frame)

        # Break on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Download the model file if you don't have it
    import requests
    url = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/1/efficientdet_lite0.tflite"
    with open('efficientdet_lite0.tflite', 'wb') as f:
        f.write(requests.get(url).content)
    
    main()