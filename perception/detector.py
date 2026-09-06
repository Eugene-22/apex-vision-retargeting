"""MediaPipe Tasks hand tracking for BGR camera/video frames."""
from pathlib import Path
import numpy as np
from .landmarks import HandObservation, validate_side, hand_frame


class HandDetector:
    def __init__(self, model_path, hand_side="right", min_confidence=0.5,
                 input_mirrored=False):
        self.hand_side = validate_side(hand_side)
        if not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence must be in [0, 1]")
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"Missing hand model: {model_path}. Run python -m perception.download_model")
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise ImportError('Install vision dependencies: pip install -e ".[vision]"') from exc
        self._mp = mp
        self.input_mirrored = input_mirrored
        self.min_confidence = min_confidence
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence)
        self._detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self._last_timestamp = -1.0
        self._last_ms = -1

    def detect(self, bgr_frame, timestamp: float) -> HandObservation | None:
        """None means no usable requested hand in this frame (never stale data).

        Tasks Hand Landmarker labels physical hands in unmirrored input (unlike
        the old Solutions Hands convention). Mirrored input is unflipped for
        inference; image landmarks are mapped back onto the original frame.
        """
        frame = np.asarray(bgr_frame)
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8 or not frame.size:
            raise ValueError("Expected a nonempty uint8 BGR image of shape (H, W, 3)")
        if not np.isfinite(timestamp) or timestamp < 0 or timestamp <= self._last_timestamp:
            raise ValueError("Frame timestamps must be finite, nonnegative and strictly increasing")
        rgb = frame[:, :, ::-1]
        if self.input_mirrored:
            rgb = rgb[:, ::-1]
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                               data=np.ascontiguousarray(rgb))
        timestamp_ms = max(self._last_ms + 1, int(timestamp * 1000))
        result = self._detector.detect_for_video(image, timestamp_ms)
        self._last_timestamp, self._last_ms = timestamp, timestamp_ms
        return self._observation(result, timestamp)

    def _observation(self, result, timestamp):
        candidates = []
        for categories, world, normalized in zip(result.handedness,
                result.hand_world_landmarks, result.hand_landmarks):
            if not categories:
                continue
            category = categories[0]
            if category.category_name.lower() != self.hand_side or category.score < self.min_confidence:
                continue
            kp = np.array([[p.x, p.y, p.z] for p in world], dtype=float)
            image = np.array([[p.x, p.y, p.z] for p in normalized], dtype=float)
            if self.input_mirrored:
                # Keep world geometry unmirrored, only map the overlay back.
                image[:, 0] = 1 - image[:, 0]
            try:
                hand_frame(kp)
                observation = HandObservation(kp, self.hand_side, timestamp,
                                              float(category.score), image)
            except ValueError:
                continue
            candidates.append(observation)
        return max(candidates, key=lambda x: x.score, default=None)

    def close(self):
        if self._detector is not None:
            self._detector.close()
            self._detector = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
