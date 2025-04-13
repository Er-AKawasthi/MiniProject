import cv2
import numpy as np
from FaceClusteringLibrary import FaceClusterUtility

def find_image_in_video(image_path, video_path):
    # Load the image
    image_to_find = cv2.imread(image_path)

    # Create VideoCapture object
    video_capture = cv2.VideoCapture(video_path)

    # Initialize FaceClusterUtility
    face_cluster_utility = FaceClusterUtility("encodings.pickle")

    # Loop through each frame of the video
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Perform face clustering on the frame
        label_ids = face_cluster_utility.ClusterFrame(frame)

        # Check if the image is found in the frame
        found = False
        # Example: You may use template matching or other techniques to find the image
        # Here we just check if the dimensions match for demonstration purpose
        if frame.shape[:2] == image_to_find.shape[:2]:
            if np.array_equal(frame, image_to_find):
                found = True

        if found:
            # Print the location where the image is found
            timestamp = video_capture.get(cv2.CAP_PROP_POS_MSEC)
            print(f"Image found at timestamp: {timestamp} milliseconds")
            # You can also print the frame number using cv2.CAP_PROP_POS_FRAMES

    # Release the VideoCapture object
    video_capture.release()

if __name__ == "__main__":
    # Specify the image and video paths
    image_path = "image_to_find.jpg"
    video_path = "footage"

    # Call the function to find the image in the video
    find_image_in_video(image_path, video_path)
