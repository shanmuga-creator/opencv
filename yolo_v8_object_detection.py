import cv2
from ultralytics import YOLO

def run_yolo_object_detection():
    # Load the YOLOv8 model
    model = YOLO('yolov8n.pt')  # Use 'yolov8n.pt' or any other version like 'yolov8s.pt', 'yolov8m.pt', etc.

    # Open the camera (use '0' for built-in webcam or '1' for external USB camera)
    cap = cv2.VideoCapture(1)  # Change '1' to '0' if needed

    if not cap.isOpened():
        print("Cannot open USB Camera")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        # Perform object detection
        results = model(frame)

        # Visualize results on the frame
        annotated_frame = results[0].plot()

        # Display the resulting frame
        cv2.imshow('YOLOv8 Object Detection', annotated_frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_yolo_object_detection()
