import cv2
from ultralytics import YOLO
from pupil_apriltags import Detector
from functions_apriltag_lib import (parse_args, detect_and_draw_tags, initialize_video_writer,
                                    get_video, fetch_frame, homography_matrix, TagTracker)


def main():
    # Parse arguments
    args = parse_args()

    # Define the URL of the camera stream
    url = "http://192.168.1.44:8080/shot.jpg"

    # Fetch the first frame to determine the frame size
    frame = fetch_frame(url)

    fps = 25
    delay = int(1000/fps)

    # Initialize the AprilTag detector with adjusted parameters
    detector = Detector(
        families=args["type"],
        nthreads=4,
        quad_decimate=2.0,
        quad_sigma=0.0,
        refine_edges=True,
        decode_sharpening=0.25
    )
    print("AprilTag detector initialized.")

    # Initialize the VideoWriter to save the output video
    out = initialize_video_writer(frame=frame, filename='output_video.avi', fps=fps)
    print("VideoWriter initialized.")

    # Initialize the tracked tags dictionary
    tracked_tags = {}

    # Calculate homography
    homography = homography_matrix()
    if homography is None:
        print("Error: Homography could not be computed.")
        return
    else:
        print("Homography matrix computed.")

    # Initialize the tag tracker
    tag_tracker = TagTracker(tracker_type='csrt')  # You can choose 'kcf' or 'csrt'

    # Optional: Resize display window
    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    # print("Display window created.")

    # Load the YOLOv8 model
    model = YOLO('yolo11s-seg.pt')  # Ensure the model is in the current directory or specify the full path
    print("YOLOv8 model loaded.")

    # Loop over the frames from the video
    frame_count = 0
    while True:
        print("Entering main loop...")
        # Fetch the frame from the smartphone camera URL
        frame = fetch_frame(url)
        if frame is None:
            break  # End of video
        # else:
            # print(f"Frame {frame_count} read successfully.")

        # Detect and draw AprilTags with tracking
        frame, tracked_tags = detect_and_draw_tags(
            frame, detector, tracked_tags, homography, tag_tracker, model)
        # print(f"AprilTags detected and drawn on frame {frame_count}.")

        # Write the frame to the video file
        out.write(frame)
        # print(f"Frame {frame_count} written to output video.")

        # Resize the frame for display if needed
        display_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # Show the output frame
        cv2.imshow("Frame", display_frame)
        # print(f"Frame {frame_count} displayed.")
        key = cv2.waitKey(delay) & 0xFF

        # Increment frame count
        frame_count += 1

        # Press Esc key to exit
        if key == 27:
            print("Escape key pressed. Exiting.")
            break

    # Cleanup
    cv2.destroyAllWindows()
    out.release()
    print("Resources released. Program finished.")


if __name__ == "__main__":
    main()


# WITH RECORDED VIDEO

'''
def main():
    # Parse arguments
    args = parse_args()

    # Initialize the AprilTag detector with adjusted parameters
    detector = Detector(
        families=args["type"],
        nthreads=4,
        quad_decimate=2.0,
        quad_sigma=0.0,
        refine_edges=True,
        decode_sharpening=0.25
    )
    print("AprilTag detector initialized.")

    # Define the path to the video file
    video_path = "path_to_your_video/{filename}.avi"
    video_path = video_path.format(filename="output_test_apriltag")
    print(f"Video path: {video_path}")

    # Video capture object
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return
    else:
        print("Video file opened successfully.")

    # Fetch the first frame to determine the frame size
    frame = get_video(cap)
    if frame is None:
        print("Error: Cannot read frame from video file.")
        return
    else:
        print("First frame read successfully.")

    # Get the FPS of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 25.0  # Default value
    delay = int(1000 / fps)
    print(f"Video FPS: {fps}")

    # Initialize the VideoWriter to save the output video
    out = initialize_video_writer(video_path=video_path, frame=frame, filename='output_video.avi', fps=fps)
    print("VideoWriter initialized.")

    # Initialize the tracked tags dictionary
    tracked_tags = {}

    # Calculate homography
    homography = homography_matrix()
    if homography is None:
        print("Error: Homography could not be computed.")
        return
    else:
        print("Homography matrix computed.")

    # Initialize the tag tracker
    tag_tracker = TagTracker(tracker_type='csrt')  # You can choose 'kcf' or 'csrt'

    # Optional: Resize display window
    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    # print("Display window created.")

    # Load the YOLOv8 model
    model = YOLO('yolo11s-seg.pt')  # Ensure the model is in the current directory or specify the full path
    print("YOLOv8 model loaded.")

    # Loop over the frames from the video
    frame_count = 0
    while True:
        print("Entering main loop...")
        # Fetch the frame from the video file
        frame = get_video(cap)
        if frame is None:
            break  # End of video
        # else:
            # print(f"Frame {frame_count} read successfully.")

        # Detect and draw AprilTags with tracking
        frame, tracked_tags = detect_and_draw_tags(
            frame, detector, tracked_tags, homography, tag_tracker, model)
        # print(f"AprilTags detected and drawn on frame {frame_count}.")

        # Write the frame to the video file
        out.write(frame)
        # print(f"Frame {frame_count} written to output video.")

        # Resize the frame for display if needed
        display_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # Show the output frame
        cv2.imshow("Frame", display_frame)
        # print(f"Frame {frame_count} displayed.")
        key = cv2.waitKey(delay) & 0xFF

        # Increment frame count
        frame_count += 1

        # Press Esc key to exit
        if key == 27:
            print("Escape key pressed. Exiting.")
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    out.release()
    print("Resources released. Program finished.")


if __name__ == "__main__":
    main()
'''