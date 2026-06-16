import shutil
import subprocess
from pathlib import Path
from collections import defaultdict
from typing import Any

import cv2
import numpy as np
import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

from config import (
    OUTPUT_VIDEO_NAME,
    PREVIEW_IMAGE_NAME,
    MIN_DET_CONF_FOR_OCR,
    MIN_CROP_WIDTH,
    MIN_CROP_HEIGHT,
    SKIP_NO_ID,
    OCR_EVERY_N_FRAMES,
    MAX_OCR_PER_TRACK,
    BBOX_PAD_X_RATIO,
    BBOX_PAD_Y_RATIO,
    CROP_RESIZE_SCALE,
    FONT_CANDIDATES,
)

from yolo_service import detect_image, track_video
from ocr_client import call_ocr_api
from result_utils import (
    create_analysis_dir,
    make_base_result,
    save_result_json,
    get_best_record,
    build_province_counts,
    build_timeline_from_tracks,
    summarize_track_records,
)


DEFAULT_TEXT = "pending"


# =========================
# FONT / DRAW HELPERS
# =========================

def get_thai_font(font_size: int = 24):
    for font_path in FONT_CANDIDATES:
        if Path(font_path).exists():
            return ImageFont.truetype(str(font_path), font_size)

    print("WARNING: Thai font not found. Thai text may not render correctly.")
    return ImageFont.load_default()


THAI_FONT = get_thai_font(font_size=24)


def make_short_label(track_id: Any, best_record: dict | None, vote_count: int) -> str:
    if best_record is None:
        return f"id:{track_id} {DEFAULT_TEXT}"

    plate_number = best_record.get("plate_number") or ""

    if not plate_number:
        plate_text = best_record.get("plate_text", "")
        plate_number = plate_text.split(",")[0] if "," in plate_text else plate_text

    if not plate_number:
        plate_number = DEFAULT_TEXT

    label = f"id:{track_id} {plate_number}"

    if vote_count > 0:
        label += f" vote:{vote_count}"

    return label


def draw_thai_label(frame, x1, y1, x2, y2, label: str):
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    bbox = draw.textbbox((0, 0), label, font=THAI_FONT)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    padding_x = 8
    padding_y = 6

    label_x = x1
    label_y = max(y1 - text_h - 12, 0)

    frame_w = frame.shape[1]

    if label_x + text_w + padding_x * 2 > frame_w:
        label_x = max(frame_w - text_w - padding_x * 2, 0)

    rect_xy = [
        label_x,
        label_y,
        label_x + text_w + padding_x * 2,
        label_y + text_h + padding_y * 2,
    ]

    draw.rectangle(rect_xy, fill=(0, 255, 0))
    draw.text(
        (label_x + padding_x, label_y + padding_y),
        label,
        font=THAI_FONT,
        fill=(0, 0, 0),
    )

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# =========================
# VIDEO CONVERT HELPER
# =========================

def convert_video_to_h264(input_path: Path, output_path: Path) -> bool:
    """
    Convert OpenCV mp4v output to browser-compatible H.264 MP4.

    Streamlit/browser video player works better with:
    - libx264
    - yuv420p
    - faststart
    """

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        print(f"Video convert failed: input not found: {input_path}")
        return False

    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        command = [
            ffmpeg_exe,
            "-y",
            "-i", str(input_path),
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_path),
        ]

        subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

        return output_path.exists() and output_path.stat().st_size > 0

    except Exception as e:
        print(f"Video convert to H.264 failed: {e}")
        return False


# =========================
# BBOX / CROP HELPERS
# =========================

def clamp_bbox(x1, y1, x2, y2, frame_w, frame_h):
    x1 = max(0, min(int(x1), frame_w - 1))
    x2 = max(0, min(int(x2), frame_w - 1))
    y1 = max(0, min(int(y1), frame_h - 1))
    y2 = max(0, min(int(y2), frame_h - 1))
    return x1, y1, x2, y2


def pad_bbox(x1, y1, x2, y2, frame_w, frame_h):
    box_w = x2 - x1
    box_h = y2 - y1

    pad_x = int(box_w * BBOX_PAD_X_RATIO)
    pad_y = int(box_h * BBOX_PAD_Y_RATIO)

    return clamp_bbox(
        x1 - pad_x,
        y1 - pad_y,
        x2 + pad_x,
        y2 + pad_y,
        frame_w,
        frame_h,
    )


def preprocess_crop_for_ocr(crop):
    if crop is None or crop.size == 0:
        return None

    if CROP_RESIZE_SCALE != 1.0:
        crop = cv2.resize(
            crop,
            None,
            fx=CROP_RESIZE_SCALE,
            fy=CROP_RESIZE_SCALE,
            interpolation=cv2.INTER_CUBIC,
        )

    return crop


def should_send_to_ocr(track_id, det_conf, crop_w, crop_h, ocr_attempts):
    if SKIP_NO_ID and isinstance(track_id, str) and track_id.startswith("no_id"):
        return False, "skip no_id"

    if det_conf < MIN_DET_CONF_FOR_OCR:
        return False, f"low det_conf={det_conf:.3f}"

    if crop_w < MIN_CROP_WIDTH or crop_h < MIN_CROP_HEIGHT:
        return False, f"small crop={crop_w}x{crop_h}"

    if ocr_attempts >= MAX_OCR_PER_TRACK:
        return False, "max ocr attempts"

    return True, "ok"


# =========================
# SIMPLE PSEUDO TRACKER
# =========================

def bbox_iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union


def bbox_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def center_distance(box_a, box_b) -> float:
    ax, ay = bbox_center(box_a)
    bx, by = bbox_center(box_b)
    return float(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)


class PseudoTracker:
    """
    ใช้สำหรับ detection ที่ YOLO tracker ยังไม่มี track_id จริง

    เป้าหมาย:
    - ไม่ใช้ det_79_1 ที่เปลี่ยนทุก frame
    - ไม่ใช้ no_id_0 ที่ซ้ำทั้งวิดีโอ
    - ให้ pseudo_id ต่อเนื่องถ้า bbox อยู่ใกล้ตำแหน่งเดิม
    """

    def __init__(
        self,
        iou_threshold: float = 0.15,
        max_center_distance: float = 80.0,
        max_missing_frames: int = 20,
    ):
        self.iou_threshold = iou_threshold
        self.max_center_distance = max_center_distance
        self.max_missing_frames = max_missing_frames

        self.next_id = 1
        self.tracks: dict[str, dict[str, Any]] = {}

    def assign(self, bbox: list[int], frame_idx: int) -> str:
        best_track_id = None
        best_score = -1.0

        expired = []
        for pseudo_id, data in self.tracks.items():
            last_frame = data["last_frame"]
            if frame_idx - last_frame > self.max_missing_frames:
                expired.append(pseudo_id)

        for pseudo_id in expired:
            self.tracks.pop(pseudo_id, None)

        for pseudo_id, data in self.tracks.items():
            last_bbox = data["bbox"]

            iou = bbox_iou(bbox, last_bbox)
            dist = center_distance(bbox, last_bbox)

            is_match = (
                iou >= self.iou_threshold
                or dist <= self.max_center_distance
            )

            if not is_match:
                continue

            score = iou + max(0.0, 1.0 - dist / self.max_center_distance)

            if score > best_score:
                best_score = score
                best_track_id = pseudo_id

        if best_track_id is None:
            best_track_id = f"pseudo_{self.next_id}"
            self.next_id += 1

        self.tracks[best_track_id] = {
            "bbox": bbox,
            "last_frame": frame_idx,
        }

        return best_track_id


def resolve_track_id(det: dict[str, Any], pseudo_tracker: PseudoTracker, frame_idx: int) -> Any:
    raw_track_id = det.get("track_id")

    if raw_track_id is not None:
        return raw_track_id

    bbox = det["bbox"]
    return pseudo_tracker.assign(bbox, frame_idx)


# =========================
# IMAGE PIPELINE
# =========================

def analyze_image(image_path: str | Path) -> dict[str, Any]:
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    analysis_dir = create_analysis_dir("image")
    crop_dir = analysis_dir / "crops"
    preview_path = analysis_dir / PREVIEW_IMAGE_NAME

    original_copy_path = analysis_dir / image_path.name
    shutil.copy2(image_path, original_copy_path)

    frame = cv2.imread(str(image_path))

    if frame is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    h, w = frame.shape[:2]

    detections = detect_image(image_path)

    records: list[dict[str, Any]] = []

    for det_idx, det in enumerate(detections):
        track_id = det["track_id"]
        det_conf = det["det_conf"]
        x1, y1, x2, y2 = det["bbox"]

        x1, y1, x2, y2 = clamp_bbox(x1, y1, x2, y2, w, h)

        crop_w = x2 - x1
        crop_h = y2 - y1

        can_ocr, reason = should_send_to_ocr(
            track_id=track_id,
            det_conf=det_conf,
            crop_w=crop_w,
            crop_h=crop_h,
            ocr_attempts=0,
        )

        ocr_result = None
        crop_path = None

        if can_ocr:
            x1p, y1p, x2p, y2p = pad_bbox(x1, y1, x2, y2, w, h)
            crop = frame[y1p:y2p, x1p:x2p]
            crop = preprocess_crop_for_ocr(crop)

            if crop is not None and crop.size != 0:
                crop_path = crop_dir / f"image_{det_idx}_crop.jpg"
                cv2.imwrite(str(crop_path), crop)
                ocr_result = call_ocr_api(crop_path)
        else:
            print(f"Skip OCR image det={det_idx} | reason={reason}")

        record = {
            "track_id": track_id,
            "bbox": [x1, y1, x2, y2],
            "det_conf": det_conf,
            "crop_path": str(crop_path) if crop_path else None,
            "plate_text": "",
            "plate_number": "",
            "province": None,
            "ocr_conf": 0.0,
            "status": "no_ocr_result",
        }

        if ocr_result:
            record.update({
                "plate_text": ocr_result.get("plate_text", ""),
                "plate_number": ocr_result.get("plate_number", ""),
                "province": ocr_result.get("province"),
                "ocr_conf": ocr_result.get("ocr_conf", 0.0),
                "raw": ocr_result.get("raw"),
                "status": "ok",
            })

        records.append(record)

        best_record = record if record["status"] == "ok" else None
        label = make_short_label(track_id, best_record, 1 if best_record else 0)
        frame = draw_thai_label(frame, x1, y1, x2, y2, label)

    cv2.imwrite(str(preview_path), frame)

    result = make_base_result(
        analysis_id=analysis_dir.name,
        file_type="image",
        file_name=image_path.name,
        output_path=None,
        preview_path=str(preview_path),
    )

    result["tracks"] = records
    result["detections"] = records
    result["province_counts"] = build_province_counts(records)
    result["timeline"] = []

    result_path = save_result_json(result, analysis_dir)
    result["result_path"] = str(result_path)

    return result


# =========================
# VIDEO PIPELINE
# =========================

def analyze_video(video_path: str | Path) -> dict[str, Any]:
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    analysis_dir = create_analysis_dir("video")
    crop_dir = analysis_dir / "crops"

    raw_video_path = analysis_dir / "output_raw.mp4"
    output_video_path = analysis_dir / OUTPUT_VIDEO_NAME

    original_copy_path = analysis_dir / video_path.name
    shutil.copy2(video_path, original_copy_path)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps == 0:
        fps = 30

    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(raw_video_path),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError("Cannot open VideoWriter. Check output path or codec.")

    track_records = defaultdict(lambda: {
        "detections": 0,
        "ocr_attempts": 0,
        "plates": [],
        "skipped": [],
    })

    pseudo_tracker = PseudoTracker(
        iou_threshold=0.15,
        max_center_distance=90.0,
        max_missing_frames=20,
    )

    print("Start analyze video...")

    for item in track_video(video_path):
        frame_idx = item["frame_idx"]
        frame = item["frame"]
        detections = item["detections"]

        h, w = frame.shape[:2]

        for det in detections:
            track_id = resolve_track_id(det, pseudo_tracker, frame_idx)

            det_conf = det["det_conf"]
            x1, y1, x2, y2 = det["bbox"]

            x1, y1, x2, y2 = clamp_bbox(x1, y1, x2, y2, w, h)

            crop_w = x2 - x1
            crop_h = y2 - y1

            if crop_w <= 0 or crop_h <= 0:
                continue

            track_records[track_id]["detections"] += 1

            ocr_attempts = track_records[track_id]["ocr_attempts"]
            has_no_ocr = len(track_records[track_id]["plates"]) == 0

            should_ocr_by_frame = (
                has_no_ocr
                or frame_idx % OCR_EVERY_N_FRAMES == 0
            )

            if should_ocr_by_frame:
                can_ocr, reason = should_send_to_ocr(
                    track_id=track_id,
                    det_conf=det_conf,
                    crop_w=crop_w,
                    crop_h=crop_h,
                    ocr_attempts=ocr_attempts,
                )

                if not can_ocr:
                    track_records[track_id]["skipped"].append({
                        "frame": frame_idx,
                        "reason": reason,
                        "det_conf": det_conf,
                        "crop_size": f"{crop_w}x{crop_h}",
                    })

                else:
                    x1p, y1p, x2p, y2p = pad_bbox(x1, y1, x2, y2, w, h)
                    crop = frame[y1p:y2p, x1p:x2p]
                    crop = preprocess_crop_for_ocr(crop)

                    if crop is not None and crop.size != 0:
                        track_records[track_id]["ocr_attempts"] += 1
                        crop_rank = track_records[track_id]["ocr_attempts"]

                        crop_path = (
                            crop_dir
                            / f"track_{track_id}_rank_{crop_rank}_frame_{frame_idx:05d}.jpg"
                        )

                        cv2.imwrite(str(crop_path), crop)

                        ocr_result = call_ocr_api(crop_path)

                        if ocr_result and ocr_result.get("plate_text"):
                            record = {
                                "track_id": track_id,
                                "plate_text": ocr_result.get("plate_text", ""),
                                "plate_number": ocr_result.get("plate_number", ""),
                                "province": ocr_result.get("province"),
                                "ocr_conf": ocr_result.get("ocr_conf", 0.0),
                                "raw": ocr_result.get("raw"),
                                "det_conf": det_conf,
                                "frame": frame_idx,
                                "crop_path": str(crop_path),
                                "crop_size": f"{crop.shape[1]}x{crop.shape[0]}",
                            }

                            track_records[track_id]["plates"].append(record)

                            print(
                                f"[frame {frame_idx}] "
                                f"track {track_id} -> {record['plate_text']} "
                                f"| province={record['province']} "
                                f"| ocr_conf={record['ocr_conf']} "
                                f"| det_conf={det_conf:.3f}"
                            )

            best_record, vote_count = get_best_record(track_records[track_id]["plates"])
            label = make_short_label(track_id, best_record, vote_count)

            frame = draw_thai_label(
                frame,
                x1,
                y1,
                x2,
                y2,
                label,
            )

        writer.write(frame)

    writer.release()

    converted = convert_video_to_h264(raw_video_path, output_video_path)

    if not converted:
        print("WARNING: H.264 conversion failed. Using raw video as fallback.")
        shutil.copy2(raw_video_path, output_video_path)

    tracks = summarize_track_records(track_records, fps=fps)
    province_counts = build_province_counts(tracks)
    timeline = build_timeline_from_tracks(tracks)

    result = make_base_result(
        analysis_id=analysis_dir.name,
        file_type="video",
        file_name=video_path.name,
        output_path=str(output_video_path),
        preview_path=None,
    )

    result["tracks"] = tracks
    result["province_counts"] = province_counts
    result["timeline"] = timeline

    detections_result = []
    for track_id, data in track_records.items():
        for plate in data.get("plates", []):
            detections_result.append(plate)

    result["detections"] = detections_result

    result_path = save_result_json(result, analysis_dir)
    result["result_path"] = str(result_path)

    print("\n========== OUTPUT ==========")
    print(f"Raw video saved: {raw_video_path}")
    print(f"H.264 video saved: {output_video_path}")
    print(f"Result saved: {result_path}")
    print(f"Crops saved: {crop_dir}")

    return result