from pathlib import Path
import sys
import time

import cv2

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


from perception.mediapipe_hand import (
    MediaPipeHandDetector,
)

from visualization.hand_draw import (
    draw_hand_landmarks,
)

from human_hand.palm_frame import landmarks_to_numpy
from human_hand.state import HumanHandState
from perception.quality import temporal_quality

from retargeting.calibration import (
    IndexCalibration,
)

from retargeting.index_mapping import (
    map_index_angles,
)

from robot.apex_urdf import (
    ApexUrdfModel,
    DEFAULT_RIGHT_URDF,
)

from robot.virtual_index import (
    apex_index_ranges,
)


MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "hand_landmarker.task"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "hand_landmarks_live.mp4"
)

# Human index calibration (M2.3).
#
# `human` ranges are PLACEHOLDERS that must be replaced by a real
# per-user calibration (M13). The Apex target ranges are NOT stored
# here: they are read from the URDF (single source of truth, M2.4).
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "calibration"
    / "index.json"
)

CAMERA_DEVICE = "/dev/video0"

WIDTH = 1280
HEIGHT = 720
CAMERA_FPS = 30


def main() -> None:
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cap = cv2.VideoCapture(
        CAMERA_DEVICE,
        cv2.CAP_V4L2,
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera: {CAMERA_DEVICE}"
        )

    # Camera format
    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG"),
    )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        WIDTH,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        HEIGHT,
    )

    cap.set(
        cv2.CAP_PROP_FPS,
        CAMERA_FPS,
    )

    actual_width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    actual_height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    actual_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    print(
        f"Camera: "
        f"{actual_width}x{actual_height} "
        f"@ {actual_fps:.1f} FPS"
    )

    writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        cv2.VideoWriter_fourcc(*"mp4v"),
        actual_fps,
        (
            actual_width,
            actual_height,
        ),
    )

    if not writer.isOpened():
        raise RuntimeError(
            "Could not create output video"
        )

    detector = MediaPipeHandDetector(
        model_path=MODEL_PATH,
        num_hands=1,
    )

    if not CONFIG_PATH.exists():
        raise RuntimeError(
            f"Missing index retargeting config: {CONFIG_PATH}\n"
            "Create it before running teleoperation."
        )

    calibration = IndexCalibration.load(
        CONFIG_PATH
    )

    urdf_model = ApexUrdfModel(
        DEFAULT_RIGHT_URDF
    )

    apex_range = apex_index_ranges(
        urdf_model
    )

    print(
        f"Index config loaded: {CONFIG_PATH}"
    )
    print(
        f"Apex URDF loaded:   {urdf_model.path}"
    )

    start_time = time.monotonic()
    previous_time = start_time
    previous_world_landmarks = None

    frame_index = 0
    detected_frames = 0

    total_inference_ms = 0.0

    try:
        while True:
            # -------------------------
            # 1. Camera capture
            # -------------------------

            capture_start = time.perf_counter()

            ok, frame = cap.read()

            capture_ms = (
                time.perf_counter()
                - capture_start
            ) * 1000

            if not ok:
                print("Failed to read frame")
                continue

            # -------------------------
            # 2. BGR -> RGB
            # -------------------------

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            # MediaPipe VIDEO mode requires
            # monotonically increasing timestamp
            timestamp_ms = int(
                (
                    time.monotonic()
                    - start_time
                )
                * 1000
            )

            # -------------------------
            # 3. MediaPipe inference
            # -------------------------

            inference_start = (
                time.perf_counter()
            )

            result = detector.detect(
                rgb_frame,
                timestamp_ms,
            )

            inference_ms = (
                time.perf_counter()
                - inference_start
            ) * 1000

            total_inference_ms += (
                inference_ms
            )

            # -------------------------
            # 4. FPS
            # -------------------------

            current_time = time.monotonic()

            dt = (
                current_time
                - previous_time
            )

            previous_time = current_time

            if dt > 0:
                instantaneous_fps = 1.0 / dt
            else:
                instantaneous_fps = 0.0

            # -------------------------
            # 5. Hand landmarks
            # -------------------------

            hand_text = "No hand"

            if result.hand_landmarks:
                detected_frames += 1

                # --------------------------------
                # 5.1. Image landmarks
                #      用于在图像上画 21 个关键点
                # --------------------------------
                landmarks = result.hand_landmarks[0]

                draw_hand_landmarks(
                    frame,
                    landmarks,
                )

                # --------------------------------
                # 5.2. Left / Right hand
                # --------------------------------
                if result.handedness:
                    handedness = result.handedness[0][0]

                    hand_text = (
                        f"{handedness.category_name} "
                        f"{handedness.score:.2f}"
                    )
                else:
                    hand_text = "Hand"

                # Convert the MediaPipe result at the perception boundary.
                hand_state = HumanHandState.from_mediapipe_result(
                    result,
                    timestamp=current_time,
                )

                if hand_state is not None:
                    quality, temporal_valid = temporal_quality(
                        hand_state.world_landmarks,
                        previous_world_landmarks,
                        dt if previous_world_landmarks is not None else None,
                    )
                    previous_world_landmarks = hand_state.world_landmarks
                    if not temporal_valid:
                        hand_state = None
                    else:
                        index_angles = hand_state.index_angles

                    # --------------------------------
                    # 5.5. Human -> Apex index mapping
                    # --------------------------------
                    apex_target = map_index_angles(
                        index_angles,
                        calibration,
                        apex_range,
                    )

                    # --------------------------------
                    # 5.6. Virtual fingertip (FK)
                    # --------------------------------
                    tip_position = (
                        urdf_model.index_tip_position(
                            apex_target.j0_abduction,
                            apex_target.j1_mcp_flexion,
                            apex_target.j2_pip_flexion,
                        )
                    )

                    # 每 30 帧打印一次
                    if frame_index % 30 == 0:
                        print(
                            "Human | "
                            f"MCP abd: "
                            f"{np.degrees(index_angles.mcp_abduction):6.1f} deg | "
                            f"MCP flex: "
                            f"{np.degrees(index_angles.mcp_flexion):6.1f} deg | "
                            f"PIP flex: "
                            f"{np.degrees(index_angles.pip_flexion):6.1f} deg"
                        )

                        print(
                            "Apex  | "
                            f"j0 abd: "
                            f"{np.degrees(apex_target.j0_abduction):6.1f} deg | "
                            f"j1 mcp: "
                            f"{np.degrees(apex_target.j1_mcp_flexion):6.1f} deg | "
                            f"j2 pip: "
                            f"{np.degrees(apex_target.j2_pip_flexion):6.1f} deg"
                        )

                        print(
                            "Tip   | "
                            f"x: {tip_position[0] * 1000:6.1f} mm | "
                            f"y: {tip_position[1] * 1000:6.1f} mm | "
                            f"z: {tip_position[2] * 1000:6.1f} mm"
                        )

            # -------------------------
            # 6. Overlay information
            # -------------------------

            cv2.putText(
                frame,
                f"FPS: {instantaneous_fps:.1f}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"Inference: {inference_ms:.1f} ms",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"Capture: {capture_ms:.1f} ms",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                hand_text,
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # -------------------------
            # 7. Save annotated frame
            # -------------------------

            writer.write(frame)

            if frame_index % 30 == 0:
                print(
                    f"frame={frame_index:06d} "
                    f"fps={instantaneous_fps:5.1f} "
                    f"inference={inference_ms:5.1f}ms "
                    f"capture={capture_ms:5.1f}ms "
                    f"{hand_text}"
                )

            frame_index += 1

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        cap.release()
        writer.release()
        detector.close()

    # -------------------------
    # Summary
    # -------------------------

    elapsed = (
        time.monotonic()
        - start_time
    )

    if frame_index > 0:
        average_fps = (
            frame_index / elapsed
        )

        average_inference_ms = (
            total_inference_ms
            / frame_index
        )

        detection_rate = (
            detected_frames
            / frame_index
            * 100
        )
    else:
        average_fps = 0.0
        average_inference_ms = 0.0
        detection_rate = 0.0

    print()
    print("=== Summary ===")
    print(
        f"Frames:            {frame_index}"
    )
    print(
        f"Elapsed:           {elapsed:.2f} s"
    )
    print(
        f"Average FPS:       {average_fps:.2f}"
    )
    print(
        f"Average inference: "
        f"{average_inference_ms:.2f} ms"
    )
    print(
        f"Detection rate:    "
        f"{detection_rate:.1f}%"
    )
    print(
        f"Output:            {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()