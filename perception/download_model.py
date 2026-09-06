"""Explicit download; importing/running the detector never accesses the network."""
import argparse
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile, is_zipfile

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")


def main():
    parser = argparse.ArgumentParser(description="Download the MediaPipe hand model")
    parser.add_argument("--output", default="models/hand_landmarker.task")
    args = parser.parse_args()
    path = Path(args.output)
    if path.is_file():
        print(f"Model already exists: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(MODEL_URL, timeout=60) as response:
        data = response.read()
    # MediaPipe task bundles may prepend alignment bytes before the ZIP header.
    if not is_zipfile(BytesIO(data)) or len(data) < 100_000:
        raise RuntimeError("Downloaded response is not a MediaPipe task archive")
    with ZipFile(BytesIO(data)) as archive:
        if archive.testzip() is not None or not any(name.endswith(".tflite") for name in archive.namelist()):
            raise RuntimeError("Downloaded model archive failed validation")
    temporary = path.with_suffix(".download")
    temporary.write_bytes(data)
    temporary.replace(path)
    print(path)


if __name__ == "__main__":
    main()
