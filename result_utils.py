import json
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import Any

from config import OUTPUT_DIR, RESULT_JSON_NAME


def create_analysis_id(prefix: str = "analysis") -> str:
    """
    สร้าง analysis id จาก timestamp
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


def create_analysis_dir(file_type: str = "analysis") -> Path:
    """
    สร้าง folder สำหรับเก็บผลลัพธ์แต่ละครั้ง
    เช่น outputs/video_20260608_153000/
    """
    analysis_id = create_analysis_id(file_type)
    analysis_dir = OUTPUT_DIR / analysis_id

    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "crops").mkdir(parents=True, exist_ok=True)

    return analysis_dir


def save_result_json(result: dict[str, Any], analysis_dir: Path) -> Path:
    """
    save result เป็น result.json
    """
    analysis_dir = Path(analysis_dir)
    result_path = analysis_dir / RESULT_JSON_NAME

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result_path


def load_result_json(result_path: Path) -> dict[str, Any] | None:
    """
    load result.json
    """
    result_path = Path(result_path)

    if not result_path.exists():
        return None

    with open(result_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_history() -> list[dict[str, Any]]:
    """
    อ่าน history จาก outputs/*/result.json
    """
    history: list[dict[str, Any]] = []

    if not OUTPUT_DIR.exists():
        return history

    for result_path in sorted(
        OUTPUT_DIR.glob(f"*/{RESULT_JSON_NAME}"),
        reverse=True
    ):
        data = load_result_json(result_path)

        if not data:
            continue

        data["_result_path"] = str(result_path)
        history.append(data)

    return history


def get_best_record(records: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, int]:
    """
    เลือก OCR result ที่ดีที่สุดจาก vote
    ถ้า plate_text ซ้ำกัน ให้เลือกตัวที่ vote มากสุด
    แล้วใช้ confidence สูงสุดในกลุ่มนั้น
    """
    if not records:
        return None, 0

    plate_texts = [
        r.get("plate_text")
        for r in records
        if r.get("plate_text")
    ]

    if not plate_texts:
        return None, 0

    counter = Counter(plate_texts)
    best_plate_text, vote_count = counter.most_common(1)[0]

    candidates = [
        r for r in records
        if r.get("plate_text") == best_plate_text
    ]

    best_record = sorted(
        candidates,
        key=lambda x: (
            x.get("ocr_conf") or x.get("api_conf") or 0,
            x.get("det_conf") or 0
        ),
        reverse=True
    )[0]

    return best_record, vote_count


def build_province_counts(tracks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    นับจำนวนจังหวัดจาก tracks
    """
    provinces = [
        t.get("province") or t.get("plate_province")
        for t in tracks
        if t.get("province") or t.get("plate_province")
    ]

    counter = Counter(provinces)

    return [
        {
            "province": province,
            "count": count
        }
        for province, count in counter.most_common()
    ]


def build_timeline_from_tracks(tracks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    สร้าง timeline จาก tracks
    """
    timeline: list[dict[str, Any]] = []

    for track in tracks:
        time_sec = track.get("first_seen_sec")

        if time_sec is None:
            continue

        timeline.append({
            "time_sec": time_sec,
            "track_id": track.get("track_id"),
            "plate_text": track.get("plate_text"),
            "plate_number": track.get("plate_number"),
            "province": track.get("province"),
            "confidence": track.get("best_confidence"),
        })

    timeline.sort(key=lambda x: x.get("time_sec", 0))

    return timeline


def format_seconds(seconds: float | int | None) -> str:
    """
    แปลงวินาทีเป็น mm:ss
    """
    if seconds is None:
        return "--:--"

    seconds = int(seconds)
    minutes = seconds // 60
    sec = seconds % 60

    return f"{minutes:02d}:{sec:02d}"


def make_base_result(
    analysis_id: str,
    file_type: str,
    file_name: str,
    output_path: str | None = None,
    preview_path: str | None = None,
) -> dict[str, Any]:
    """
    สร้าง result structure กลาง
    """
    return {
        "analysis_id": analysis_id,
        "file_type": file_type,
        "file_name": file_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_path": output_path,
        "preview_path": preview_path,
        "province_counts": [],
        "tracks": [],
        "detections": [],
        "timeline": [],
    }


def summarize_track_records(
    track_records: dict[Any, dict[str, Any]],
    fps: float = 30.0,
) -> list[dict[str, Any]]:
    """
    แปลง track_records จาก pipeline ให้เป็น tracks สำหรับ result.json
    """
    tracks: list[dict[str, Any]] = []

    for track_id, data in track_records.items():
        plates = data.get("plates", [])
        detections = data.get("detections", 0)

        best_record, vote_count = get_best_record(plates)

        if not best_record:
            tracks.append({
                "track_id": track_id,
                "plate_text": "",
                "plate_number": "",
                "province": None,
                "vote_count": 0,
                "best_confidence": 0.0,
                "detections": detections,
                "first_seen_sec": None,
                "last_seen_sec": None,
                "status": "no_ocr_result",
            })
            continue

        frames = [
            p.get("frame")
            for p in plates
            if p.get("frame") is not None
        ]

        first_frame = min(frames) if frames else None
        last_frame = max(frames) if frames else None

        first_seen_sec = round(first_frame / fps, 2) if first_frame is not None else None
        last_seen_sec = round(last_frame / fps, 2) if last_frame is not None else None

        best_conf = (
            best_record.get("ocr_conf")
            or best_record.get("api_conf")
            or 0.0
        )

        tracks.append({
            "track_id": track_id,
            "plate_text": best_record.get("plate_text", ""),
            "plate_number": best_record.get("plate_number", ""),
            "province": best_record.get("province") or best_record.get("plate_province"),
            "vote_count": vote_count,
            "best_confidence": best_conf,
            "detections": detections,
            "first_seen_sec": first_seen_sec,
            "last_seen_sec": last_seen_sec,
            "status": "ok",
        })

    return tracks