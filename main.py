import cv2
import os

# Path to the video file
video_path = "E:/android/archive\SCVD/videos/violence video cleaned\V83.mp4"

# Create a directory to save preprocessed frames
output_dir = "preprocessed_frames"
os.makedirs(output_dir, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Codec and VideoWriter for saving preprocessed video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video_path = "E:/android/archive\SCVD"
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply preprocessing steps here
    # Example: Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Save the preprocessed frame
    frame_filename = os.path.join(output_dir, f"preprocessed_frame_{cap.get(cv2.CAP_PROP_POS_FRAMES)}.jpg")
    cv2.imwrite(frame_filename, gray_frame)

    # Write preprocessed frame to output video
    out.write(gray_frame)

    # Display the original and preprocessed frames (optional)
    cv2.imshow("Original Frame", frame)
    cv2.imshow("Preprocessed Frame", gray_frame)

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

# Release video capture and writer objects
cap.release()
out.release()

# Destroy all OpenCV windows
cv2.destroyAllWindows()