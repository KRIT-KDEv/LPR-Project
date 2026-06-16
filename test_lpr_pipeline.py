from pathlib import Path

from ocr_client import check_ocr_server
from lpr_pipeline import analyze_image, analyze_video


def test_image():
    image_path = Path("test_images/test_car1.png")
    result = analyze_image(image_path)

    print("\nIMAGE RESULT:")
    print("analysis_id:", result["analysis_id"])
    print("preview_path:", result["preview_path"])
    print("province_counts:", result["province_counts"])
    print("tracks:", result["tracks"])


def test_video():
    video_path = Path("test_videos/test_car_video2.mp4")
    result = analyze_video(video_path)

    print("\nVIDEO RESULT:")
    print("analysis_id:", result["analysis_id"])
    print("output_path:", result["output_path"])
    print("province_counts:", result["province_counts"])
    print("timeline:", result["timeline"])
    print("tracks:", result["tracks"])


if __name__ == "__main__":
    if not check_ocr_server():
        print("OCR Server is not ready.")
        raise SystemExit

    # เริ่มจาก image ก่อน
    test_image()

    # ถ้า image ผ่านแล้ว ค่อยเปิดบรรทัดนี้
    test_video()