import cv2
import numpy as np
import mediapipe as mp


BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="pose_landmarker_lite.task"
    ),
    running_mode=VisionRunningMode.VIDEO
)


pose = PoseLandmarker.create_from_options(options)


def analyze_motion(video_path, detected_objects):

    if "person" not in detected_objects:
        return "non-human motion"

    cap = cv2.VideoCapture(video_path)

    positions = []

    frame_idx = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        timestamp_ms = frame_idx * 33

        frame_idx += 1

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        results = pose.detect_for_video(
            mp_image,
            timestamp_ms
        )

        if results.pose_landmarks:

            lm = results.pose_landmarks[0]

            left = lm[23]
            right = lm[24]

            avg = (left.y + right.y) / 2

            positions.append(avg)

    cap.release()

    if len(positions) < 2:
        return "minimal movement"

    arr = np.array(positions)

    movement = arr[-1] - arr[0]

    if movement < -0.02:
        return "moving upward"

    elif movement > 0.02:
        return "moving downward"

    else:
        return "mostly stationary"