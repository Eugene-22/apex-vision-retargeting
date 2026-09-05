from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class MediaPipeHandDetector:
    """
    MediaPipe Hand Landmarker wrapper.

    Input:
        RGB image: np.ndarray [H, W, 3]

    Output:
        MediaPipe HandLandmarkerResult
    """

    def __init__(
        self,
        model_path: str | Path,
        num_hands: int = 1,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe model not found: {model_path}"
            )

        base_options = python.BaseOptions(
            model_asset_path=str(model_path)
        )

        options = vision.HandLandmarkerOptions(
            base_options=base_options,

            # 摄像头是连续视频流
            running_mode=vision.RunningMode.VIDEO,

            num_hands=num_hands,

            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self._detector = (
            vision.HandLandmarker.create_from_options(
                options
            )
        )

    def detect(
        self,
        rgb_frame: np.ndarray,
        timestamp_ms: int,
    ):
        """
        Detect hand landmarks from one RGB frame.
        """

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        return self._detector.detect_for_video(
            mp_image,
            timestamp_ms,
        )

    def close(self) -> None:
        self._detector.close()