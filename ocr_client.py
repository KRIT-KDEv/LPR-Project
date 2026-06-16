from pathlib import Path
from typing import Any

import requests

from config import OCR_API_URL, OCR_HEALTH_URL, OCR_TIMEOUT


def check_ocr_server(timeout: int = 10) -> bool:
    """
    เช็กว่า OCR local server เปิดอยู่ และ model โหลดแล้วหรือยัง
    """

    try:
        response = requests.get(OCR_HEALTH_URL, timeout=timeout)

        if response.status_code != 200:
            print(f"OCR health failed | status={response.status_code}")
            return False

        data = response.json()

        status = data.get("status")
        model_loaded = data.get("model_loaded")
        device = data.get("device")

        print(f"OCR Server status={status} | model_loaded={model_loaded} | device={device}")

        return status == "ok" and model_loaded is True

    except requests.exceptions.ConnectionError:
        print("OCR Server connection error. Please run OCR server first.")
        return False

    except requests.exceptions.Timeout:
        print("OCR Server health check timeout.")
        return False

    except Exception as e:
        print(f"OCR Server health check error: {e}")
        return False


def normalize_ocr_response(data: dict[str, Any]) -> dict[str, Any]:
    """
    แปลง response จาก OCR Server ให้ format คงที่
    เพื่อให้ pipeline ใช้งานต่อได้ง่าย
    """

    plate_text = data.get("plate_text") or ""
    plate_number = data.get("plate_number") or ""
    province = data.get("plate_province")
    confidence = data.get("confidence") or 0.0
    raw = data.get("raw")

    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.0

    return {
        "plate_text": plate_text,
        "plate_number": plate_number,
        "province": province,
        "ocr_conf": confidence,
        "raw": raw,
    }


def call_ocr_api(image_path: Path | str, timeout: int = OCR_TIMEOUT) -> dict[str, Any] | None:
    """
    ส่งรูป crop ป้ายทะเบียนไป OCR Server

    input:
        image_path: path ของ crop image

    output:
        dict:
            plate_text
            plate_number
            province
            ocr_conf
            raw

        หรือ None ถ้า OCR failed
    """

    image_path = Path(image_path)

    if not image_path.exists():
        print(f"OCR image not found: {image_path}")
        return None

    try:
        with open(image_path, "rb") as f:
            files = {"file": f}

            response = requests.post(
                OCR_API_URL,
                files=files,
                timeout=timeout,
            )

        if response.status_code != 200:
            print(
                f"OCR API failed | status={response.status_code} "
                f"| file={image_path.name}"
            )
            return None

        data = response.json()
        return normalize_ocr_response(data)

    except requests.exceptions.ConnectionError:
        print("OCR API connection error. Is OCR server running?")
        return None

    except requests.exceptions.Timeout:
        print(f"OCR API timeout | file={image_path.name}")
        return None

    except ValueError:
        print(f"OCR API returned non-JSON response | file={image_path.name}")
        return None

    except Exception as e:
        print(f"OCR API exception: {e} | file={image_path.name}")
        return None