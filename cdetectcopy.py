import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker
from tello import Tello  # Import the Tello class from your Tello SDK library

model = YOLO('best.pt')

# Initialize the Tello drone
tello = Tello()

# Connect to the Tello drone
tello.connect()

# Start video stream
tello.start_video()

# Initialize tracker
tracker = Tracker()

while True:
    # Capture frame from Tello camera
    frame = tello.read_video_frame()

    # Resize frame
    frame = cv2.resize(frame, (1020, 500))

    # Flip frame horizontally
    frame = cv2.flip(frame, 1)

    # Perform object detection
    results = model.predict(frame)

    # Extract bounding boxes and class labels
    boxes = results.xyxy[0].numpy()
    classes = results.names[results.xyxy[0][:, -1].numpy().astype(int)]

    # Update tracker
    bbox_idx = tracker.update(boxes.tolist())

    # Draw bounding boxes and class labels
    for bbox in bbox_idx:
        x1, y1, x2, y2, id = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, str(id), (x2, y2), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)

    # Display frame
    cv2.imshow("FRAME", frame)

    # Check for exit key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
tello.stop_video()
tello.land()
cv2.destroyAllWindows()
