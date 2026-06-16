from pathlib import Path
import shutil

import pandas as pd
import plotly.express as px
import streamlit as st

from config import (
    ensure_directories,
    IMAGE_UPLOAD_DIR,
    VIDEO_UPLOAD_DIR,
)
from ocr_client import check_ocr_server
from lpr_pipeline import analyze_image, analyze_video
from result_utils import load_history, format_seconds


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Local LPR Dashboard",
    page_icon="🚗",
    layout="wide",
)

ensure_directories()


# =========================
# HELPERS
# =========================

def save_uploaded_file(uploaded_file, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)

    file_path = target_dir / uploaded_file.name

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path


def show_province_chart(province_counts: list[dict]):
    if not province_counts:
        st.info("No province data.")
        return

    df = pd.DataFrame(province_counts)

    if df.empty:
        st.info("No province data.")
        return

    fig = px.bar(
        df,
        x="province",
        y="count",
        title="Province Detection Count",
        text="count",
    )

    st.plotly_chart(fig, use_container_width=True)


def show_tracks_table(tracks: list[dict]):
    if not tracks:
        st.info("No track result.")
        return

    df = pd.DataFrame(tracks)

    preferred_columns = [
        "track_id",
        "plate_text",
        "plate_number",
        "province",
        "vote_count",
        "best_confidence",
        "detections",
        "first_seen_sec",
        "last_seen_sec",
        "status",
    ]

    existing_columns = [col for col in preferred_columns if col in df.columns]

    if existing_columns:
        df = df[existing_columns]

    st.dataframe(df, use_container_width=True)


def show_timeline(timeline: list[dict]):
    if not timeline:
        st.info("No timeline data.")
        return

    st.caption(f"Timeline events: {len(timeline)}")

    with st.container(height=420, border=True):
        for item in timeline:
            time_text = format_seconds(item.get("time_sec"))
            plate_text = item.get("plate_text") or "-"
            province = item.get("province") or "-"
            track_id = item.get("track_id")

            ocr_conf = item.get("ocr_conf")
            best_confidence = item.get("best_confidence")
            confidence = ocr_conf if ocr_conf is not None else best_confidence

            conf_text = ""
            if confidence is not None:
                try:
                    conf_text = f" | conf {float(confidence):.2f}"
                except Exception:
                    conf_text = ""

            st.markdown(
                f"""
                <div style="
                    padding: 8px 4px;
                    border-bottom: 1px solid rgba(128,128,128,0.25);
                    font-size: 14px;
                    line-height: 1.5;
                ">
                    <b>{time_text}</b>
                    <span style="color: gray;"> | track </span>
                    <code>{track_id}</code>
                    <span> | {plate_text} | {province}{conf_text}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


def show_ocr_details(result: dict):
    detections = result.get("detections", [])

    if not detections:
        st.info("No OCR details.")
        return

    analysis_id = result.get("analysis_id", "unknown")

    st.caption(f"OCR records: {len(detections)}")

    df = pd.DataFrame(detections)

    # ไม่โชว์ crop_path ในตารางหลัก เพราะ path ยาวและทำให้ UI ดูยาก
    preferred_columns = [
        "track_id",
        "frame",
        "plate_text",
        "plate_number",
        "province",
        "ocr_conf",
        "det_conf",
        "raw",
        "crop_size",
    ]

    existing_columns = [col for col in preferred_columns if col in df.columns]

    st.subheader("OCR Records Table")

    if existing_columns:
        st.dataframe(
            df[existing_columns],
            use_container_width=True,
            height=320,
        )
    else:
        st.dataframe(
            df,
            use_container_width=True,
            height=320,
        )

    st.subheader("OCR Crop Preview")

    max_preview = min(len(detections), 100)

    if max_preview <= 5:
        preview_limit = max_preview
    else:
        preview_limit = st.slider(
            "Number of OCR crops to preview",
            min_value=5,
            max_value=max_preview,
            value=min(max_preview, 30),
            step=5,
            key=f"ocr_preview_limit_{analysis_id}",
        )

    visible_detections = detections[:preview_limit]

    with st.container(height=620, border=True):
        has_crop = False

        for idx, item in enumerate(visible_detections):
            crop_path = item.get("crop_path")

            if not crop_path:
                continue

            crop_path_obj = Path(crop_path)

            if not crop_path_obj.exists():
                continue

            has_crop = True

            track_id = item.get("track_id", "-")
            frame = item.get("frame", "-")
            plate_text = item.get("plate_text", "-")
            plate_number = item.get("plate_number", "-")
            province = item.get("province", "-")
            ocr_conf = item.get("ocr_conf", 0.0)
            det_conf = item.get("det_conf", 0.0)
            raw = item.get("raw", "-")
            crop_size = item.get("crop_size", "-")

            with st.expander(
                f"OCR #{idx + 1} | track {track_id} | frame {frame} | {plate_text}"
            ):
                col1, col2 = st.columns([1, 2])

                with col1:
                    st.image(
                        str(crop_path_obj),
                        caption="Plate crop sent to OCR",
                        use_container_width=True,
                    )

                with col2:
                    st.write(f"**Track ID:** `{track_id}`")
                    st.write(f"**Frame:** `{frame}`")
                    st.write(f"**Plate Text:** `{plate_text}`")
                    st.write(f"**Plate Number:** `{plate_number}`")
                    st.write(f"**Province:** `{province}`")
                    st.write(f"**OCR Confidence:** `{ocr_conf}`")
                    st.write(f"**Detection Confidence:** `{det_conf}`")
                    st.write(f"**Crop Size:** `{crop_size}`")
                    st.write(f"**Raw OCR:** `{raw}`")

        if not has_crop:
            st.info("No crop preview available.")

    if len(detections) > preview_limit:
        st.caption(
            f"Showing {preview_limit} of {len(detections)} OCR crops. "
            "Increase the slider to preview more."
        )


def show_result(result: dict):
    st.subheader("Analysis Result")

    col1, col2, col3 = st.columns(3)

    col1.metric("Type", result.get("file_type", "-"))
    col2.metric("File", result.get("file_name", "-"))
    col3.metric("Analysis ID", result.get("analysis_id", "-"))

    st.divider()

    preview_path = result.get("preview_path")
    output_path = result.get("output_path")

    if preview_path and Path(preview_path).exists():
        st.subheader("Preview Image")
        st.image(preview_path, use_container_width=True)

    if output_path and Path(output_path).exists():
        st.subheader("Output Video")
        st.video(output_path)

    st.divider()

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Province Graph")
        show_province_chart(result.get("province_counts", []))

    with right:
        st.subheader("Timeline")
        show_timeline(result.get("timeline", []))

    st.divider()

    st.subheader("Result Table")
    show_tracks_table(result.get("tracks", []))

    st.divider()

    st.subheader("OCR Details")
    show_ocr_details(result)

    with st.expander("Raw JSON"):
        st.json(result)


# =========================
# SIDEBAR
# =========================

st.sidebar.title("Local LPR Dashboard")

st.sidebar.caption("YOLO local + OCR local GPU")

if st.sidebar.button("Check OCR Server"):
    ready = check_ocr_server()
    if ready:
        st.sidebar.success("OCR Server ready")
    else:
        st.sidebar.error("OCR Server not ready")

st.sidebar.divider()

page = st.sidebar.radio(
    "Menu",
    ["Analyze", "History"],
)


# =========================
# MAIN PAGE
# =========================

st.title("🚗 Local LPR Dashboard")

if page == "Analyze":
    st.subheader("Upload Image / Video")

    file_type = st.radio(
        "Select input type",
        ["Image", "Video"],
        horizontal=True,
    )

    if file_type == "Image":
        uploaded_file = st.file_uploader(
            "Upload image",
            type=["jpg", "jpeg", "png", "webp"],
        )

        if uploaded_file:
            st.image(
                uploaded_file,
                caption="Uploaded image",
                use_container_width=True,
            )

            if st.button("Analyze Image", type="primary"):
                if not check_ocr_server():
                    st.error("OCR Server is not ready. Please run OCR Server first.")
                    st.stop()

                input_path = save_uploaded_file(uploaded_file, IMAGE_UPLOAD_DIR)

                with st.spinner("Analyzing image..."):
                    result = analyze_image(input_path)

                st.success("Image analysis completed.")
                show_result(result)

    else:
        uploaded_file = st.file_uploader(
            "Upload video",
            type=["mp4", "avi", "mov", "mkv"],
        )

        if uploaded_file:
            input_path = save_uploaded_file(uploaded_file, VIDEO_UPLOAD_DIR)

            st.video(str(input_path))

            if st.button("Analyze Video", type="primary"):
                if not check_ocr_server():
                    st.error("OCR Server is not ready. Please run OCR Server first.")
                    st.stop()

                with st.spinner("Analyzing video... This may take a while."):
                    result = analyze_video(input_path)

                st.success("Video analysis completed.")
                show_result(result)


elif page == "History":
    st.subheader("Analysis History")

    history = load_history()

    if not history:
        st.info("No analysis history yet.")
    else:
        for item in history:
            analysis_id = item.get("analysis_id", "-")
            file_name = item.get("file_name", "-")
            file_type = item.get("file_type", "-")
            created_at = item.get("created_at", "-")

            with st.expander(
                f"{analysis_id} | {file_type} | {file_name} | {created_at}"
            ):
                show_result(item)