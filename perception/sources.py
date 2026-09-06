"""Context-managed video capture and explicit JSONL observation recording."""
import json
import time
from pathlib import Path
import numpy as np
from .landmarks import HandObservation, validate_side


class VideoSource:
    def __init__(self, source=0):
        self.source = source
        self._capture = None

    def __enter__(self):
        import cv2
        self._capture = cv2.VideoCapture(self.source)
        if not self._capture.isOpened():
            self.close()
            raise RuntimeError(f"Cannot open camera/video: {self.source}")
        self._fps = self._capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(self._fps) or self._fps <= 0:
            self._fps = 30.0
        self._start = time.monotonic()
        return self

    def __iter__(self):
        if self._capture is None:
            raise RuntimeError("Use VideoSource as a context manager")
        index = 0
        while True:
            ok, frame = self._capture.read()
            if not ok:
                if isinstance(self.source, int):
                    raise RuntimeError(f"Camera {self.source} stopped returning frames")
                return
            timestamp = (time.monotonic() - self._start if isinstance(self.source, int)
                         else index / self._fps)
            yield timestamp, frame
            index += 1

    def close(self):
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __exit__(self, *_):
        self.close()


class ObservationWriter:
    """Record missing detections as null to preserve tracking gaps on replay."""
    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        self._file = self.path.open("w", encoding="utf-8")
        return self

    def write(self, timestamp, observation):
        if not np.isfinite(timestamp) or timestamp < 0:
            raise ValueError("Invalid recording timestamp")
        if observation is not None and observation.timestamp != timestamp:
            raise ValueError("Recording timestamp must match the observation")
        item = {"version": 1, "timestamp": timestamp, "hand": None}
        if observation is not None:
            item["hand"] = {"hand_side": observation.hand_side, "score": observation.score,
                            "keypoints": observation.keypoints.tolist()}
        self._file.write(json.dumps(item, allow_nan=False) + "\n")

    def __exit__(self, *_):
        self._file.close()


class ReplaySource:
    def __init__(self, path, hand_side="right"):
        self.path = Path(path)
        self.hand_side = validate_side(hand_side)

    def __iter__(self):
        previous = -1.0
        with self.path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    item = json.loads(line)
                    if item["version"] != 1:
                        raise ValueError("Unsupported recording version")
                    timestamp = float(item["timestamp"])
                    if not np.isfinite(timestamp) or timestamp < 0 or timestamp <= previous:
                        raise ValueError("Timestamps must be finite and strictly increasing")
                    hand = item["hand"]
                    observation = None if hand is None else HandObservation(
                        hand["keypoints"], hand["hand_side"], timestamp, hand["score"])
                    if observation is not None and observation.hand_side != self.hand_side:
                        observation = None
                except (ValueError, KeyError, TypeError) as exc:
                    raise ValueError(f"{self.path}:{line_number}: {exc}") from exc
                previous = timestamp
                yield timestamp, observation
