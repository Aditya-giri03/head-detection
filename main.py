import time
import socket
import mediapipe as mp
import numpy as np
import struct
import json

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2

DETECTION_RESULT = None

def save_result(
    result: vision.PoseLandmarkerResult,
    unused_output_image: mp.Image,
    timestamp_ms: int,
):
    global DETECTION_RESULT
    DETECTION_RESULT = result


# Initialize the pose landmarker model
base_options = python.BaseOptions(model_asset_path="./pose_landmarker_lite.task")
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_poses=5,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_segmentation_masks=False,
    result_callback=save_result,
)
detector = vision.PoseLandmarker.create_from_options(options)


def preprocess_image(img, w, h):
    # Convert image bytes to OpenCV image
    # img = np.array(Image.open(io.BytesIO(image_bytes)).convert(mode="RGB"))
    # img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW
    img = np.frombuffer(img, np.uint8).reshape(h, w, 3)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    detector.detect_async(mp_image, time.time_ns() // 1_000_000)
    nose_coords = []

    if DETECTION_RESULT is not None:
        for pose_landmarks in DETECTION_RESULT.pose_landmarks:
            temp = [pose_landmarks[0].x, pose_landmarks[0].y, pose_landmarks[0].z]
            nose_coords.append((temp[0],temp[1]))

        return nose_coords

    return None


config = {
    "process_id": "head",
    "server_address": "/tmp/gesurease.sock",
}


def run():
    img_width_bytes = sock.recv(4)
    img_height_bytes = sock.recv(4)
    data_len_bytes = sock.recv(4)
    if len(data_len_bytes) == 0:
        print("Connection closed, exiting...")
        exit(1)

    img_width = struct.unpack("!I", img_width_bytes)[0]
    img_height = struct.unpack("!I", img_height_bytes)[0]
    data_len = struct.unpack("!I", data_len_bytes)[0]

    img = sock.recv(data_len)
    while len(img) < data_len:
        img += sock.recv(data_len - len(img))


    # print(img)
    nose_coords = preprocess_image(img, img_width, img_height)

    if nose_coords is not None:
        json_data = []
        for i in nose_coords:
            dict = {"nose_x": i[0]*img_width, "nose_y": i[1]*img_height}
            json_data.append(dict)

        json_response = json.dumps({"prediction": json_data})
        sock.sendall(struct.pack("!I", len(json_response)))
        sock.sendall(json_response.encode())
    else:
        json_response = json.dumps(
            {"prediction": [{"nose_x": 0.0, "nose_y": 0.0}]}
        )
        sock.sendall(struct.pack("!I", len(json_response)))
        sock.sendall(json_response.encode())
    time.sleep(0.1)

if __name__ == "__main__":
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(config["server_address"])
    sock.setblocking(True)

    # Send the process identifier to the Rust server
    sock.sendall(config["process_id"].encode())

    while True:
        run()
