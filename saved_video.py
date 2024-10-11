import cv2
from ultralytics import YOLO

def run_yolo_object_detection():
    # Load the YOLOv8 model
    model = YOLO('yolov8n.pt')

    # Open the camera (use '0' for built-in webcam or '1' for external USB camera)
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Cannot open USB Camera")
        return

    # Set the desired frame size (width and height)
    frame_width = 1280  # Set width to 1280 pixels
    frame_height = 720  # Set height to 720 pixels
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    # Define the codec and create VideoWriter object to save video
    # 'XVID' is a common codec, but you can also use 'MJPG', 'MP4V', etc.
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for .avi files
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (frame_width, frame_height))  # 20.0 is the frame rate

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

        # Save the annotated frame to the video file
        out.write(annotated_frame)

        # Display the resulting frame
        cv2.imshow('YOLOv8 Object Detection', annotated_frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture and VideoWriter objects, and close OpenCV windows
    cap.release()
    out.release()  # Release the video writer
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_yolo_object_detection()
