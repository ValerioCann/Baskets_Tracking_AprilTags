import unittest
import cv2
import requests
import numpy as np
from functions import get_aruco_dict_and_params
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)


class TestAprilTags(unittest.TestCase):

    def setUp(self):
        """
        Setup method that runs before each test. Fetches and processes the frame.
        """
        url = "http://192.168.1.44:8080/shot.jpg"
        img_resp = requests.get(url)
        self.img_resp = img_resp
        self.img = cv2.imdecode(np.array(bytearray(img_resp.content), dtype=np.uint8), -1)
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        aruco_dict, aruco_params = get_aruco_dict_and_params("DICT_APRILTAG_36h11")
        aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        self.corners, self.ids, self.rejected = aruco_detector.detectMarkers(self.gray)

    def tearDown(self):
        """
        Tear down method to clean up after tests. Resets the variables.
        """
        self.img_resp = None
        self.img = None
        self.gray = None
        self.corners = None
        self.ids = None
        self.rejected = None
        print("Tear down completed.")

    def test_camera_connection(self):
        """
        TC001 - Verifies the successful connection to the smartphone's camera by checking the HTTP response status code.
        Expected outcome: Response status code 200
        """
        self.assertEqual(self.img_resp.status_code, 200, "Cannot connect to the camera")
        print(Fore.GREEN + f"test_camera_connection passed: Camera connected successfully. Status code: {self.img_resp.status_code}")

    def test_grayscale_conversion(self):
        """
        TC002 - Check if the gray conversion is performed correctly.
        Expected outcome: The grayscale image should have two dimensions (height, width).
        """
        self.assertEqual(len(self.gray.shape), 2, "Grayscale conversion failed")
        print(Fore.GREEN + "test_grayscale_conversion passed: Grayscale conversion successful.")

    def test_apriltag_detection(self):
        """
        TC003 - Verifies if the AprilTags in the captured image are detected successfully.
        Expected outcome: The ids variable should not be None, and it should contain at least one ID.
        """
        self.assertIsNotNone(self.ids, "No AprilTags detected")
        self.assertGreater(len(self.ids), 0, "No AprilTags detected")
        print(Fore.GREEN + "test_apriltag_detection passed: AprilTags detected successfully.")

    def test_bounding_box_drawing(self):
        """
        TC004 - Verifies that the bounding boxes around the detected AprilTags are correctly generated.
        Expected outcome: The corner points should be tuples of two elements (x, y)
        """
        r = self.corners[0].reshape((4, 2))
        ptA, ptB, ptC, ptD = r
        self.assertIsInstance(ptA, np.ndarray, "Bounding box points not correctly generated")
        self.assertEqual(ptA.shape, (2,), "Bounding box points not correctly generated")
        print(Fore.GREEN + "test_bounding_box_drawing passed: Bounding box points generated successfully.")

    def test_tag_id_annotation(self):
        """
        TC005 - Verifies that the Tag IDs are correctly annotated on the image.
        Expected outcome: The text size (width and height) for the Tag ID should be greater than 0.
        """
        tagID = self.ids[0]
        text = f"ID: {tagID}"
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        self.assertGreater(w, 0, "Tag ID text size calculation failed")
        self.assertGreater(h, 0, "Tag ID text size calculation failed")
        print(Fore.GREEN + "test_tag_id_annotation passed: Tag ID annotation successful.")

    def test_video_output(self):
        """
        TC006 - Checks if the video file is opened successfully for writing
        Expected outcome: The video writer should successfully open the output file for writing.
        """
        output_filename = 'output_video.avi'
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        fps = 6.0
        out = cv2.VideoWriter(output_filename, fourcc, fps, (1000, 1800))
        self.assertTrue(out.isOpened(), "Failed to open video file for writing")
        print(Fore.GREEN + "test_video_output passed: Video file opened successfully.")


if __name__ == "__main__":
    # Tests executed in alphabetical order
    unittest.main(verbosity=2)

    # If we want to manage the order
    #suite = unittest.TestSuite()
    #suite.addTest(TestAprilTags("test_camera_connection"))
    #suite.addTest(TestAprilTags("test_grayscale_conversion"))
    #suite.addTest(TestAprilTags("test_apriltag_detection"))
    #suite.addTest(TestAprilTags("test_bounding_box_drawing"))
    #suite.addTest(TestAprilTags("test_tag_id_annotation"))
    #suite.addTest(TestAprilTags("test_video_output"))

    #runner = unittest.TextTestRunner()
    #runner.run(suite)
