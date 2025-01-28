import cv2
import os

def video_to_images(video_path, output_folder, frame_rate=1):
    """
    Convert a video to a sequence of images.

    :param video_path: Path to the input video file.
    :param output_folder: Folder to save the output images.
    :param frame_rate: Number of frames to skip (1 = save every frame).
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Total frames: {total_frames}, FPS: {fps}")

    frame_count = 0
    saved_frame_count = 0

    while True:
        # Read the next frame
        ret, frame = cap.read()
        if not ret:
            break  # Exit the loop if no more frames

        # Save the frame at the specified frame rate
        if frame_count % frame_rate == 0:
            image_path = os.path.join(output_folder, f"frame_{saved_frame_count:05d}.jpg")
            cv2.imwrite(image_path, frame)
            saved_frame_count += 1

        frame_count += 1

    # Release the video capture object
    cap.release()
    print(f"Saved {saved_frame_count} frames to {output_folder}")

# Example usage
video_path = "vid.mp4"  # Replace with your video file path
output_folder = "output_images"  # Folder to save the images
frame_rate = 10  # Save every 10th frame (adjust as needed)

video_to_images(video_path, output_folder, frame_rate)