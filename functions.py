import cv2
import numpy as np
import argparse
import requests


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--type", type=str, default="tag36h11", help="Type of AprilTag to detect")
    args, unknown = ap.parse_known_args()
    return vars(args)


def fetch_frame(url):
    """
    Fetches a frame from the specified URL for live stream.
    Parameters:
    - url (str): The URL to fetch the image from.
    Returns:
    - frame (np.ndarray): The image/frame fetched from the stream.
    """
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    frame = cv2.imdecode(img_arr, -1)
    return frame


def get_video(cap):
    ret, frame = cap.read()
    if not ret:
        return None
    return frame


def homography_matrix():
    # Define the source and destination points for homography
    src_points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    dst_points = np.array([[100, 100], [200, 80], [220, 200], [120, 220]], dtype=np.float32)
    H, status = cv2.findHomography(src_points, dst_points)
    if H is not None:
        return H
    else:
        return None


def transform_coordinates(image_coord, homography):
    if homography is None or homography.shape != (3, 3):
        print("Error : Invalid Homography Matrix.")
        return None
    if not isinstance(image_coord, (list, tuple, np.ndarray)) or len(image_coord) != 2:
        print("Error : Invalid image coordinates.")
        return None
    try:
        pts = np.array([[image_coord]], dtype='float32')  # Shape (1, 1, 2)
        pts_reel = cv2.perspectiveTransform(pts, homography)
        return pts_reel[0][0]
    except cv2.error as e:
        print(f"Error while transforming coordinates : {e}")
        return None


def initialize_video_writer(video_path, frame, filename='output_video.avi', fps=30.0):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_size = (frame.shape[1], frame.shape[0])  # (width, height)
    out = cv2.VideoWriter(filename, fourcc, fps, frame_size)
    return out


class TagTracker:
    def __init__(self, tracker_type='CSRT'):
        OPENCV_OBJECT_TRACKERS = {
            "csrt": cv2.TrackerCSRT.create,
            "kcf": cv2.TrackerKCF.create,
            "mil": cv2.TrackerMIL.create,
            "mosse": cv2.legacy.TrackerMOSSE.create,
        }
        self.tracker_type = tracker_type.lower()
        if self.tracker_type not in OPENCV_OBJECT_TRACKERS:
            raise ValueError(f"Tracker type '{tracker_type}' not supported.")
        self.trackers = {}
        self.OPENCV_OBJECT_TRACKERS = OPENCV_OBJECT_TRACKERS

    def init_tracker(self, frame, bbox, tag_id):
        tracker_creator = self.OPENCV_OBJECT_TRACKERS[self.tracker_type]
        tracker = tracker_creator()
        tracker.init(frame, bbox)
        self.trackers[tag_id] = tracker

    def update_tracker_position(self, tag_id, frame, bbox):
        # Re-initialize the tracker with the new bounding box
        self.init_tracker(frame, bbox, tag_id)

    def update_trackers(self, frame):
        predicted_positions = {}
        for tag_id, tracker in list(self.trackers.items()):
            success, bbox = tracker.update(frame)
            if success:
                predicted_positions[tag_id] = bbox
            else:
                print(f"Tracker lost for tag {tag_id}, removing tracker.")
                del self.trackers[tag_id]
        return predicted_positions


def detect_and_draw_tags(frame, detector, tracked_tags, homography, tag_tracker, model):
    # Convert to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Tags & ID detection
    detections = detector.detect(gray)
    detected_ids = []
    tag_centers = {}
    if detections:
        for detection in detections:
            markerID = detection.tag_id
            detected_ids.append(markerID)

            # Corners of the detected tag
            corners = detection.corners  # Shape: (4, 2)
            marker_corner = corners.astype(int)

            # Draw the contour of the tag using the corners
            cv2.polylines(frame, [marker_corner], isClosed=True, color=(255, 255, 0), thickness=2)

            # Calculate the center of the tag
            cx = int(detection.center[0])
            cy = int(detection.center[1])
            tag_centers[markerID] = (cx, cy)

            # Transform the coordinates of the center in the real plane
            real_coord = transform_coordinates([cx, cy], homography)

            # Store the current center position of the tag
            tracked_tags[markerID] = real_coord

            # Draw the center (x, y) of the AprilTag
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            # Text to display
            tag_info = f"ID: {markerID}"

            # Draw the text
            cv2.putText(frame, tag_info, (cx, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Object segmentation with YOLOv8
    results = model(frame)

    # Keep track of which tags have updated trackers
    updated_trackers = set()

    # Iterate over segmented objects
    for result in results:
        masks = result.masks  # Get the masks
        boxes = result.boxes  # Bounding boxes

        if masks is None:
            continue  # No masks detected

        # Iterate over each detected object
        for i in range(len(boxes)):
            # Get the mask at the size of the original image
            mask = masks.data[i].cpu().numpy().astype(np.uint8)  # Shape: (height, width)

            # Verify mask and frame dimensions
            if mask.shape != frame.shape[:2]:
                # Resize the mask if necessary
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

            # Check if the tag center is within the mask
            for tag_id, (cx, cy) in tag_centers.items():
                # Ensure cx and cy are within the mask bounds
                if 0 <= cy < mask.shape[0] and 0 <= cx < mask.shape[1]:
                    if mask[cy, cx]:  # If the tag center pixel is within the mask
                        # Get the axis-aligned bounding box of the mask
                        y_indices, x_indices = np.where(mask == 1)
                        x_min_mask, x_max_mask = x_indices.min(), x_indices.max()
                        y_min_mask, y_max_mask = y_indices.min(), y_indices.max()
                        bbox_obj = (x_min_mask, y_min_mask, x_max_mask - x_min_mask, y_max_mask - y_min_mask)

                        # Validate the bounding box dimensions
                        if bbox_obj[2] > 0 and bbox_obj[3] > 0:
                            # Initialize or update the tracker for this object
                            if tag_id not in tag_tracker.trackers:
                                tag_tracker.init_tracker(frame, bbox_obj, tag_id)
                                print(f"Initialized tracker for tag {tag_id}.")
                            else:
                                # Update the tracker with the new position
                                tag_tracker.update_tracker_position(tag_id, frame, bbox_obj)
                                print(f"Updated tracker for tag {tag_id}.")

                            updated_trackers.add(tag_id)

                            # Draw the axis-aligned bounding box of the object (mask)
                            top_left = (x_min_mask, y_min_mask)
                            bottom_right = (x_max_mask, y_max_mask)
                            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

                            # Optionally, draw the center of the object
                            obj_cx = int((x_min_mask + x_max_mask) / 2)
                            obj_cy = int((y_min_mask + y_max_mask) / 2)
                            # cv2.circle(frame, (obj_cx, obj_cy), 4, (0, 255, 0), -1)

                            # Update tracked_tags with the object's position
                            real_coord = transform_coordinates([obj_cx, obj_cy], homography)
                            tracked_tags[tag_id] = real_coord

                            break  # Exit the loop once the tag is associated with an object
                        else:
                            print(f"Invalid bounding box for tag {tag_id}, skipping tracker initialization.")
                else:
                    continue

    # Update trackers for tags not detected or not updated
    predicted_positions = tag_tracker.update_trackers(frame)

    for tag_id, bbox in predicted_positions.items():
        # Draw the tracking bounding box for all tracked tags
        x, y, w, h = [int(v) for v in bbox]
        top_left = (x, y)
        bottom_right = (x + w, y + h)
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

        # Calculate the center of the bounding box
        cx = x + w // 2
        cy = y + h // 2

        # Transform the center coordinates to the real plane
        real_coord = transform_coordinates([cx, cy], homography)

        # Update tracked_tags with the predicted positions
        tracked_tags[tag_id] = real_coord

        # Optionally, draw the predicted center
        # cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

        # Display the ID (and coordinates if needed)
        if tag_id in detected_ids:
            tag_info = f"ID: {tag_id} (detected)"  # coordinates : {real_coord}
        else:
            tag_info = f"ID: {tag_id} (tracked)"  # coordinates : {real_coord}
        cv2.putText(frame, tag_info, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Return the frame and the updated tracked_tags
    return frame, tracked_tags
