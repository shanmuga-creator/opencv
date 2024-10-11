import cv2
from ultralytics import YOLO

class YOLOv8RealTime:
    def __init__(self, model_paph='yolov8n.pt', webcam_index=0):
        # Load YOLOv8 model
        self.model = YOLO(model_path)
        # Set webcam index
        self.webcam_index = webcam_index

    def detect_objects(self):
        # Initialize webcam capture
        cap = cv2.VideoCapture(self.webcam_index)

        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            if not ret:
                print("Error: Could not read frame.")
                break

            # Perform YOLOv8 object detection
            results = self.model(frame)

            # Draw the results on the frame
            annotated_frame = results[0].plot()

            # Display the frame with annotations
            cv2.imshow('YOLOv8 Real-Time Detection', annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the webcam and close windows
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Initialize YOLOv8 object detection
    detector = YOLOv8RealTime(model_path='yolov8n.pt', webcam_index=0)
    # Start real-time detection
    detector.detect_objects()
