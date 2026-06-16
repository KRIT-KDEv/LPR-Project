from pathlib import Path

from yolo_service import detect_image, track_video


def test_image():
    image_path = Path("test_images/test_car1.png")

    detections = detect_image(image_path)

    print("\nImage detections:")
    for det in detections:
        print(det)


def test_video_first_frames():
    video_path = Path("test_videos/test_car_video2.mp4")

    print("\nVideo tracking first frames:")

    for item in track_video(video_path):
        frame_idx = item["frame_idx"]
        detections = item["detections"]

        if detections:
            print(f"frame={frame_idx} | detections={detections}")

        if frame_idx >= 30:
            break


if __name__ == "__main__":
    test_image()
    test_video_first_frames()