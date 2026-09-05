from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "camera_test.jpg"


def main() -> None:
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cap = cv2.VideoCapture(
        "/dev/video0",
        cv2.CAP_V4L2,
    )

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open /dev/video0"
        )

    # 使用摄像头支持的 MJPG 模式
    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG"),
    )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720,
    )

    cap.set(
        cv2.CAP_PROP_FPS,
        30,
    )

    # 注意：set() 只是请求，所以必须读取实际生效参数
    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    fourcc_value = int(
        cap.get(cv2.CAP_PROP_FOURCC)
    )

    fourcc = "".join(
        chr((fourcc_value >> (8 * i)) & 0xFF)
        for i in range(4)
    )

    print("Camera opened")
    print(f"Resolution : {width} x {height}")
    print(f"FPS        : {fps}")
    print(f"Pixel fmt  : {fourcc}")

    frame = None

    # 丢弃前 20 帧，让自动曝光/白平衡稍微稳定
    for _ in range(20):
        ok, frame = cap.read()

        if not ok:
            cap.release()
            raise RuntimeError(
                "Failed to read camera frame"
            )

    cap.release()

    if frame is None:
        raise RuntimeError(
            "Camera returned no frame"
        )

    success = cv2.imwrite(
        str(OUTPUT_PATH),
        frame,
    )

    if not success:
        raise RuntimeError(
            f"Failed to save {OUTPUT_PATH}"
        )

    print(f"Frame shape: {frame.shape}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()