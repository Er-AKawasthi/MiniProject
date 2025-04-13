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

    # Generate the frames from the recorded footage
    framesGenerator = FramesGenerator("recorded_footage.avi")
    framesGenerator.GenerateFrames("Frames")

    # Rest of your code remains unchanged
    # ...

    # Continue with the rest of your code
    CurrentPath = os.getcwd()
    FramesDirectory = "Frames"
    FramesDirectoryPath = os.path.join(CurrentPath, FramesDirectory)
    EncodingsFolder = "Encodings"
    EncodingsFolderPath = os.path.join(CurrentPath, EncodingsFolder)

    if os.path.exists(EncodingsFolderPath):
        shutil.rmtree(EncodingsFolderPath, ignore_errors=True)
        time.sleep(0.5)
    os.makedirs(EncodingsFolderPath)

    pipeline = Pipeline(
        FramesProvider("Files source", sourcePath=FramesDirectoryPath) |
        FaceEncoder("Encode faces") |
        DatastoreManager("Store encoding",
                         encodingsOutputPath=EncodingsFolderPath),
        n_threads=3, quiet=True)

    pbar = TqdmUpdate()
    pipeline.run(update_callback=pbar.update)

    print()
    print('[INFO] Encodings extracted')

    # Merge all the encodings pickle files into one
    CurrentPath = os.getcwd()
    EncodingsInputDirectory = "Encodings"
    EncodingsInputDirectoryPath = os.path.join(
        CurrentPath, EncodingsInputDirectory)

    OutputEncodingPickleFilename = "encodings.pickle"

    if os.path.exists(OutputEncodingPickleFilename):
        os.remove(OutputEncodingPickleFilename)

    picklesListCollator = PicklesListCollator(
        EncodingsInputDirectoryPath)
    picklesListCollator.GeneratePickle(
        OutputEncodingPickleFilename)

    # To manage any delay in file writing
    time.sleep(0.5)

    # Start clustering process and generate
    # output images with annotations
    EncodingPickleFilePath = "encodings.pickle"

    faceClusterUtility = FaceClusterUtility(EncodingPickleFilePath)
    faceImageGenerator = FaceImageGenerator(EncodingPickleFilePath)

    labelIDs = faceClusterUtility.Cluster()
    faceImageGenerator.GenerateImages(
        labelIDs, "ClusteredFaces", "Montage")
