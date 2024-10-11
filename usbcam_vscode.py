import cv2

# Use '1' if '0' doesn't work (depends on USB port and number of cameras)
cap = cv2.VideoCapture(1)  # Use '1' for USB camera, '0' for built-in camera

if not cap.isOpened():
    print("Cannot open USB Camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Display the resulting frame
    cv2.imshow('USB Camera', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
