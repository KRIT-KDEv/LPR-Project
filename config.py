from pathlib import Path

# =========================
# BASE PATH
# =========================

BASE_DIR = Path(__file__).resolve().parent

# =========================
# MODEL PATH
# =========================

YOLO_MODEL_PATH = BASE_DIR / "models" / "best.pt"

# =========================
# OCR SERVER
# =========================

OCR_API_URL = "http://127.0.0.1:8002/ocr/upload"
OCR_HEALTH_URL = "http://127.0.0.1:8002/health"

# =========================
# STORAGE PATH
# =========================

UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

IMAGE_UPLOAD_DIR = UPLOAD_DIR / "images"
VIDEO_UPLOAD_DIR = UPLOAD_DIR / "videos"

# =========================
# YOLO SETTINGS
# =========================

CONF_THRES = 0.25
IMG_SIZE = 960
TRACKER = "bytetrack.yaml"

# =========================
# OCR SETTINGS
# =========================

OCR_EVERY_N_FRAMES = 10
MAX_OCR_PER_TRACK = 5
OCR_TIMEOUT = 120

# =========================
# CROP FILTER
# =========================

SKIP_NO_ID = False

MIN_DET_CONF_FOR_OCR = 0.30
MIN_CROP_WIDTH = 50
MIN_CROP_HEIGHT = 20

# =========================
# CROP ENHANCEMENT
# =========================

BBOX_PAD_X_RATIO = 0.15
BBOX_PAD_Y_RATIO = 0.25
CROP_RESIZE_SCALE = 2.0

# =========================
# OUTPUT SETTINGS
# =========================

OUTPUT_VIDEO_NAME = "output_tracked_video.mp4"
PREVIEW_IMAGE_NAME = "preview_image.jpg"
RESULT_JSON_NAME = "result.json"

# =========================
# FONT SETTINGS
# =========================

FONT_CANDIDATES = [
    BASE_DIR / "fonts" / "NotoSansThai-Regular.ttf",
    Path(r"C:\Windows\Fonts\tahoma.ttf"),
    Path(r"C:\Windows\Fonts\LeelawUI.ttf"),
    Path(r"C:\Windows\Fonts\arial.ttf"),
]


# =========================
# INIT FOLDERS
# =========================

def ensure_directories() -> None:
    """
    สร้าง folder หลักที่ระบบต้องใช้
    """
    folders = [
        UPLOAD_DIR,
        OUTPUT_DIR,
        IMAGE_UPLOAD_DIR,
        VIDEO_UPLOAD_DIR,
        BASE_DIR / "models",
    ]

    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)


def validate_config() -> None:
    """
    เช็ก path สำคัญก่อนเริ่มระบบ
    """
    if not YOLO_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YOLO model not found: {YOLO_MODEL_PATH}\n"
            "กรุณาวาง best.pt ไว้ที่ models/best.pt"
        )