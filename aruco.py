import cv2 as cv
import cv2.aruco as aruco
import os

video_path = '**VIDEO_PATH**'
print(f"Using video file path: {video_path}")

if not os.path.exists(video_path):
    print(f"Error: File {video_path} does not exist.")
    exit(1)

cap = cv.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit(1)
else:
    print(f"Successfully opened video file {video_path}")

dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_100)
parameters = cv.aruco.DetectorParameters()
detector = cv.aruco.ArucoDetector(dictionary, parameters)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from video or end of video reached")
        break
    
    # Convert to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # Detect markers
    markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(gray)
    
    if markerIds is not None:
        print(f"Detected markers: {markerIds.flatten()}")
    else:
        print("No markers detected ")

    # Draw markers
    frame = cv.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
    
    # Display the frame
    cv.imshow('Aruco Markers', frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()