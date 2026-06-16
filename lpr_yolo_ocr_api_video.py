import cv2
import requests
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont


# =========================
# CONFIG
# =========================

YOLO_MODEL_PATH = "models/best.pt"
VIDEO_PATH = "test_videos/test_car_video2.mp4"
API_URL = "http://127.0.0.1:8002/ocr/upload"

OUTPUT_DIR = Path("api_video_ocr")
CROP_DIR = OUTPUT_DIR / "crops"
OUTPUT_VIDEO_PATH = OUTPUT_DIR / "output_tracked_video.mp4"

CROP_DIR.mkdir(parents=True, exist_ok=True)

CONF_THRES = 0.25
IMG_SIZE = 960

OCR_EVERY_N_FRAMES = 10
MAX_OCR_PER_TRACK = 5

DEFAULT_TEXT = "pending"


# =========================
# FONT CONFIG
# =========================

def get_thai_font(font_size=24):
    """
    ใช้ font ที่รองรับภาษาไทย
    - ถ้ามี fonts/NotoSansThai-Regular.ttf จะใช้ตัวนี้ก่อน
    - ถ้าไม่มี จะลองใช้ font Windows
    """

    candidate_fonts = [
        "fonts/NotoSansThai-Regular.ttf",
        r"C:\Windows\Fonts\tahoma.ttf",
        r"C:\Windows\Fonts\LeelawUI.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ]

    for font_path in candidate_fonts:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, font_size)

    print("WARNING: Thai font not found. Thai text may not render correctly.")
    return ImageFont.load_default()


THAI_FONT = get_thai_font(font_size=24)


# =========================
# HELPER FUNCTIONS
# =========================

def call_ocr_api(image_path: Path):
    try:
        with open(image_path, "rb") as f:
            files = {"file": f}
            response = requests.post(API_URL, files=files, timeout=60)

        if response.status_code != 200:
            print(f"OCR API Error Status: {response.status_code}")
            print(response.text)
            return None

        return response.json()

    except Exception as e:
        print(f"OCR API Exception: {e}")
        return None


def get_best_record(plates):
    """
    เลือกผล OCR ที่ดีที่สุดจาก vote
    คืนค่า:
    - best_record
    - vote_count
    """

    if not plates:
        return None, 0

    plate_texts = [
        p["plate_text"]
        for p in plates
        if p.get("plate_text")
    ]

    if not plate_texts:
        return None, 0

    counter = Counter(plate_texts)
    best_plate_text, vote_count = counter.most_common(1)[0]

    candidates = [
        p for p in plates
        if p.get("plate_text") == best_plate_text
    ]

    best_record = sorted(
        candidates,
        key=lambda x: (
            x.get("api_conf") or 0,
            x.get("det_conf") or 0
        ),
        reverse=True
    )[0]

    return best_record, vote_count


def make_short_label(track_id, best_record, vote_count):
    """
    Label บนวิดีโอให้สั้น ไม่ลากยาวเต็มจอ
    ใช้ plate_number เป็นหลัก ไม่เอาจังหวัดยาว ๆ มาแสดงบนวิดีโอ
    """

    if best_record is None:
        return f"id:{track_id} {DEFAULT_TEXT}"

    plate_number = best_record.get("plate_number")

    if not plate_number:
        plate_text = best_record.get("plate_text", "")

        if "," in plate_text:
            plate_number = plate_text.split(",")[0]
        else:
            plate_number = plate_text

    if not plate_number:
        plate_number = DEFAULT_TEXT

    label = f"id:{track_id} {plate_number}"

    if vote_count > 0:
        label += f" vote:{vote_count}"

    return label


def draw_thai_label(frame, x1, y1, x2, y2, label):
    """
    วาด bounding box + Thai label ด้วย PIL
    แก้ปัญหา cv2.putText แสดงไทยเป็น ??????
    """

    # วาด box ด้วย OpenCV ได้ปกติ
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # แปลง BGR -> RGB สำหรับ PIL
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # คำนวณขนาดข้อความ
    bbox = draw.textbbox((0, 0), label, font=THAI_FONT)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    padding_x = 8
    padding_y = 6

    label_x = x1
    label_y = max(y1 - text_h - 12, 0)

    # กัน label เกินขอบขวา
    frame_w = frame.shape[1]
    if label_x + text_w + padding_x * 2 > frame_w:
        label_x = max(frame_w - text_w - padding_x * 2, 0)

    # พื้นหลังข้อความ
    rect_xy = [
        label_x,
        label_y,
        label_x + text_w + padding_x * 2,
        label_y + text_h + padding_y * 2
    ]

    draw.rectangle(rect_xy, fill=(0, 255, 0))

    # ข้อความ
    draw.text(
        (label_x + padding_x, label_y + padding_y),
        label,
        font=THAI_FONT,
        fill=(0, 0, 0)
    )

    # แปลงกลับ RGB -> BGR
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# =========================
# MAIN
# =========================

model = YOLO(YOLO_MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps == 0:
    fps = 30

cap.release()

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO_PATH),
    fourcc,
    fps,
    (width, height)
)

if not writer.isOpened():
    raise RuntimeError("Cannot open VideoWriter. Check output path or codec.")

track_records = defaultdict(lambda: {
    "detections": 0,
    "ocr_attempts": 0,
    "plates": []
})

frame_idx = 0

print("Start YOLO Local Tracking + OCR API + Save Video...")

stream = model.track(
    source=VIDEO_PATH,
    conf=CONF_THRES,
    imgsz=IMG_SIZE,
    tracker="bytetrack.yaml",
    persist=True,
    stream=True,
    verbose=False
)

for result in stream:
    frame = result.orig_img.copy()

    if result.boxes is not None and len(result.boxes) > 0:
        for box_idx, box in enumerate(result.boxes):

            if box.id is None:
                track_id = f"no_id_{box_idx}"
            else:
                track_id = int(box.id[0].item())

            det_conf = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            h, w = frame.shape[:2]

            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w - 1))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h - 1))

            track_records[track_id]["detections"] += 1

            # =========================
            # OCR every N frames
            # =========================

            should_ocr = (
                frame_idx % OCR_EVERY_N_FRAMES == 0
                and track_records[track_id]["ocr_attempts"] < MAX_OCR_PER_TRACK
            )

            if should_ocr:
                crop = frame[y1:y2, x1:x2]

                if crop.size != 0:
                    track_records[track_id]["ocr_attempts"] += 1

                    crop_rank = track_records[track_id]["ocr_attempts"]

                    crop_path = (
                        CROP_DIR
                        / f"track_{track_id}_rank_{crop_rank}_frame_{frame_idx:05d}.jpg"
                    )

                    cv2.imwrite(str(crop_path), crop)

                    api_data = call_ocr_api(crop_path)

                    if api_data:
                        plate_text = api_data.get("plate_text")
                        plate_number = api_data.get("plate_number")
                        plate_province = api_data.get("plate_province")
                        api_conf = api_data.get("confidence")
                        raw = api_data.get("raw")

                        if plate_text:
                            record = {
                                "plate_text": plate_text,
                                "plate_number": plate_number,
                                "plate_province": plate_province,
                                "api_conf": api_conf,
                                "raw": raw,
                                "det_conf": det_conf,
                                "frame": frame_idx,
                                "crop_path": str(crop_path)
                            }

                            track_records[track_id]["plates"].append(record)

                            print(
                                f"[frame {frame_idx}] "
                                f"track {track_id} -> {plate_text} "
                                f"| plate_number={plate_number} "
                                f"| province={plate_province} "
                                f"| api_conf={api_conf} "
                                f"| det_conf={det_conf:.3f}"
                            )

            # =========================
            # Draw current best result
            # =========================

            best_record, vote_count = get_best_record(
                track_records[track_id]["plates"]
            )

            label = make_short_label(track_id, best_record, vote_count)

            frame = draw_thai_label(
                frame,
                x1,
                y1,
                x2,
                y2,
                label
            )

    writer.write(frame)
    frame_idx += 1

writer.release()

print("\n========== SUMMARY ==========")

for track_id, data in track_records.items():
    detections = data["detections"]
    ocr_attempts = data["ocr_attempts"]
    plates = data["plates"]

    print(f"\nTrack ID: {track_id} | detections: {detections} | crops: {ocr_attempts}")

    if not plates:
        print("RESULT: No OCR result")
        continue

    ranked = sorted(
        plates,
        key=lambda x: (
            x.get("api_conf") or 0,
            x.get("det_conf") or 0
        ),
        reverse=True
    )

    plate_counter = Counter([
        p["plate_text"]
        for p in plates
        if p.get("plate_text")
    ])

    best_vote_plate, vote_count = plate_counter.most_common(1)[0]
    best_single = ranked[0]["plate_text"]

    for idx, p in enumerate(ranked, start=1):
        crop_name = Path(p["crop_path"]).name

        print(
            f"  rank_{idx}_{crop_name} -> "
            f"'{p['plate_text']}' "
            f"| number={p.get('plate_number')} "
            f"| province={p.get('plate_province')} "
            f"| api_conf={p.get('api_conf')} "
            f"| det_conf={p.get('det_conf'):.3f}"
        )

    print(
        f"RESULT: {best_vote_plate} "
        f"| vote count: {vote_count} "
        f"| best single: {best_single}"
    )

print("\n========== OUTPUT ==========")
print(f"Video saved: {OUTPUT_VIDEO_PATH}")
print(f"Crops saved: {CROP_DIR}")