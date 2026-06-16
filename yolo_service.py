from pathlib import Path
from typing import Any, Generator

from ultralytics import YOLO

from config import (
    YOLO_MODEL_PATH,
    CONF_THRES,
    IMG_SIZE,
    TRACKER,
)


_yolo_model: YOLO | None = None


def get_yolo_model() -> YOLO:
    """
    Load YOLO model once and reuse it.
    """

    global _yolo_model

    if _yolo_model is None:
        model_path = Path(YOLO_MODEL_PATH)

        if not model_path.exists():
            raise FileNotFoundError(f"YOLO model not found: {model_path}")

        _yolo_model = YOLO(str(model_path))

    return _yolo_model


def normalize_box(
    box: Any,
    box_idx: int,
    frame_idx: int | None = None,
) -> dict[str, Any]:
    """
    Convert Ultralytics box object into common detection format.

    Important:
    If tracker has no real ID, return track_id=None.
    Do not create det_{frame}_{box} here.
    lpr_pipeline.py will assign pseudo track IDs.
    """

    if box.id is None:
        track_id = None
    else:
        track_id = int(box.id[0].item())

    det_conf = float(box.conf[0])
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

    detection = {
        "track_id": track_id,
        "box_idx": box_idx,
        "bbox": [int(x1), int(y1), int(x2), int(y2)],
        "det_conf": det_conf,
    }

    if frame_idx is not None:
        detection["frame_idx"] = frame_idx

    return detection


def detect_image(image_path: str | Path) -> list[dict[str, Any]]:
    """
    Detect license plates from one image.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = get_yolo_model()

    results = model.predict(
        source=str(image_path),
        conf=CONF_THRES,
        imgsz=IMG_SIZE,
        verbose=False,
    )

    detections: list[dict[str, Any]] = []

    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            continue

        for box_idx, box in enumerate(result.boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            det_conf = float(box.conf[0])

            detections.append(
                {
                    "track_id": f"image_{box_idx}",
                    "box_idx": box_idx,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "det_conf": det_conf,
                }
            )

    return detections


def track_video(video_path: str | Path) -> Generator[dict[str, Any], None, None]:
    """
    Track license plates in video.

    Yields:
        {
            "frame_idx": int,
            "frame": np.ndarray,
            "detections": [
                {
                    "track_id": int | None,
                    "box_idx": int,
                    "bbox": [x1, y1, x2, y2],
                    "det_conf": float,
                    "frame_idx": int
                }
            ]
        }
    """

    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    model = get_yolo_model()

    stream = model.track(
        source=str(video_path),
        conf=CONF_THRES,
        imgsz=IMG_SIZE,
        tracker=TRACKER,
        persist=True,
        stream=True,
        verbose=False,
    )

    frame_idx = 0

    for result in stream:
        frame = result.orig_img.copy()
        detections: list[dict[str, Any]] = []

        if result.boxes is not None and len(result.boxes) > 0:
            for box_idx, box in enumerate(result.boxes):
                detections.append(
                    normalize_box(
                        box=box,
                        box_idx=box_idx,
                        frame_idx=frame_idx,
                    )
                )

        yield {
            "frame_idx": frame_idx,
            "frame": frame,
            "detections": detections,
        }

        frame_idx += 1