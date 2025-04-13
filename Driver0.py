# Importing necessary libraries
from FaceClusteringLibrary import *
import cv2

if __name__ == "__main__":
    # Open camera and record footage
    camera = cv2.VideoCapture(0)  # 0 corresponds to the default camera, you can change it if needed

    # Create a VideoWriter object to save the footage
    fps = 30  # frames per second
    resolution = (640, 480)  # adjust the resolution as needed
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter("recorded_footage.avi", fourcc, fps, resolution)

    # Capture frames from the camera and record them
    while True:
        ret, frame = camera.read()
        if not ret:
            break

        # Write the frame to the video file
        video_writer.write(frame)

        # Display the frame
        cv2.imshow("Camera", frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and video writer
    camera.release()
    video_writer.release()
    cv2.destroyAllWindows()

    # Continue with the rest of your code
    # ...

    # Generate the frames from the recorded footage
    framesGenerator = FramesGenerator("recorded_footage.avi")
    framesGenerator.GenerateFrames("Frames")
    # Rest of your code remains unchanged
