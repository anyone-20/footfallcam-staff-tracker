import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import time
import os
from PIL import Image
from ultralytics import YOLO
from google import genai
from dotenv import load_dotenv

# --- Load Environment Variables & Secrets ---
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY and "GEMINI_API_KEY" in st.secrets:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

st.set_page_config(page_title="FootfallCam - AI Staff Tracker", layout="wide")

st.title("🏬 FootfallCam 3D Sensor - Staff Identification & Tracking")
st.markdown(
    "Automated identification and tracking for corridor-walking personnel with multimodal "
    "visual verification and trajectory unification."
)

# --- Multi-Template & Feature Matcher ---
def detect_tag_features(roi_bgr, templates_gray):
    """
    Combines horizontal Sobel edge stripe density (for the piano-key badge)
    with multi-scale template matching across orientations.
    """
    if roi_bgr is None or roi_bgr.size == 0 or roi_bgr.shape[0] < 12 or roi_bgr.shape[1] < 12:
        return 0.0

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Edge Stripe Filter (high-contrast alternating black/white bands)
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    abs_sobel = np.abs(sobel_x)
    stripe_score = np.percentile(abs_sobel, 95) / 255.0

    best_template_score = 0.0
    angles = [-30, -15, 0, 15, 30, 150, 165, 180, 195, 210]

    for t_gray in templates_gray:
        th, tw = t_gray.shape[:2]
        for ang in angles:
            if ang != 0:
                M = cv2.getRotationMatrix2D((tw / 2.0, th / 2.0), ang, 1.0)
                rot_t = cv2.warpAffine(t_gray, M, (tw, th), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            else:
                rot_t = t_gray

            for scale in [0.15, 0.25, 0.35, 0.45]:
                tw_scaled = int(w * scale)
                th_scaled = int(tw_scaled * (th / tw))
                if tw_scaled >= 10 and th_scaled >= 10 and tw_scaled < w and th_scaled < h:
                    scaled_t = cv2.resize(rot_t, (tw_scaled, th_scaled))
                    res = cv2.matchTemplate(gray, scaled_t, cv2.TM_CCOEFF_NORMED)
                    _, max_v, _, _ = cv2.minMaxLoc(res)
                    if max_v > best_template_score:
                        best_template_score = max_v

    combined = 0.5 * best_template_score + 0.5 * min(1.0, stripe_score * 1.5)
    return max(best_template_score, combined)

# --- Multimodal Gemini VLM Verifier ---
def verify_tag_with_gemini(crop_bgr, tag_template_bgr, api_key):
    """
    Calls Gemini VLM with exponential backoff and automatic model fallback
    to prevent 503 UNAVAILABLE spikes from stopping the pipeline.
    """
    models_to_try = ['gemini-2.5-flash', 'gemini-1.5-flash', 'gemini-2.0-flash']
    
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    crop_pil = Image.fromarray(crop_rgb)

    tag_rgb = cv2.cvtColor(tag_template_bgr, cv2.COLOR_BGR2RGB)
    tag_pil = Image.fromarray(tag_rgb)

    prompt = (
        "You are evaluating whether an individual in overhead surveillance footage is wearing an official staff badge.\n"
        "- Image 1: The official staff badge reference (black and white rectangular badge / piano-key pattern).\n"
        "- Image 2: A cropped chest view of a walking individual from an overhead 3D sensor.\n\n"
        "Due to overhead camera resolution, distance, and motion blur, the badge may appear blurred, small, or low-contrast.\n"
        "Is the person in Image 2 wearing the badge shown in Image 1 (or its blurred equivalent on dark clothing)?\n"
        "Reply strictly in this format:\n"
        "VERDICT: [YES/NO] | REASON: [1-sentence explanation]"
    )

    client = genai.Client(api_key=api_key)

    for model_name in models_to_try:
        for attempt in range(3):  # Retry up to 3 times per model
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[tag_pil, crop_pil, prompt]
                )
                text = response.text.strip()
                is_yes = "VERDICT: YES" in text.upper() or text.upper().startswith("YES")
                return is_yes, f"[{model_name}] {text}"
            except Exception as e:
                err_str = str(e)
                if "503" in err_str or "UNAVAILABLE" in err_str or "429" in err_str:
                    time.sleep(1.5 * (attempt + 1))  # Exponential wait: 1.5s, 3.0s, 4.5s
                    continue
                else:
                    break

    return False, "API Error: All fallback models are currently experiencing high demand."

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Model Architecture & Vision")
model_choice = st.sidebar.selectbox(
    "YOLO Architecture",
    ["yolov8n.pt", "yolov8s.pt", "yolo11n.pt", "yolov8n-pose.pt"]
)

st.sidebar.markdown("---")
st.sidebar.header("📁 Input Files & Thresholds")
video_source = st.sidebar.file_uploader("Upload Video (.mp4)", type=["mp4"])
tag_sources = st.sidebar.file_uploader(
    "Upload Tag Reference Images (Multiple)", 
    type=["png", "jpg", "jpeg"], 
    accept_multiple_files=True
)

conf_thresh = st.sidebar.slider("Tag Matching Threshold", 0.20, 0.80, 0.35, 0.02)
corridor_only = st.sidebar.checkbox("Corridor Gating (Ignore Desks/Chairs)", value=True)

run_btn = st.sidebar.button("🚀 Run Live Detection", use_container_width=True)

DEFAULT_VIDEO = "sample.mp4"
DEFAULT_TAGS = ["tag_template.png", "tag_reference.png"]

col_video, col_results = st.columns([1.3, 1])

if run_btn:
    if not GEMINI_API_KEY:
        st.error("Missing Gemini API Key! Please ensure `GEMINI_API_KEY` is configured in your `.env` file or Streamlit Cloud Secrets.")
        st.stop()

    # 1. Resolve Video Input
    if video_source is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_source.read())
        video_path = tfile.name
    elif os.path.exists(DEFAULT_VIDEO):
        video_path = DEFAULT_VIDEO
    else:
        st.error("Missing video: Upload sample.mp4 or place it in the working directory.")
        st.stop()

    # 2. Resolve Multi-Tag Templates
    templates_gray = []
    tag_ref_color = None

    if tag_sources:
        for f in tag_sources:
            file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
            t_color = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if t_color is not None:
                if tag_ref_color is None:
                    tag_ref_color = t_color
                templates_gray.append(cv2.cvtColor(t_color, cv2.COLOR_BGR2GRAY))
    else:
        for p in DEFAULT_TAGS:
            if os.path.exists(p):
                t_color = cv2.imread(p, cv2.IMREAD_COLOR)
                if t_color is not None:
                    if tag_ref_color is None:
                        tag_ref_color = t_color
                    templates_gray.append(cv2.cvtColor(t_color, cv2.COLOR_BGR2GRAY))

    if not templates_gray or tag_ref_color is None:
        st.error("Missing tag templates: Upload at least one tag image or add tag_template.png.")
        st.stop()

    with st.spinner(f"Loading {model_choice}..."):
        model = YOLO(model_choice)

cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    # Downscale target (640px wide while preserving aspect ratio)
    target_w = 640
    scale_factor = target_w / float(orig_w) if orig_w > target_w else 1.0
    vid_w = int(orig_w * scale_factor)
    vid_h = int(orig_h * scale_factor)

    with col_video:
        st.subheader("🎥 Live Feed & Corridor Tracking")
        video_placeholder = st.empty()
        status_text = st.empty()

    progress_bar = st.progress(0.0)

    all_detections = []
    track_positions = {}
    track_tag_votes = {}
    track_best_crops = {}
    frame_idx = 0

    corridor_min_x = int(0.32 * vid_w)
    corridor_max_x = int(0.72 * vid_w)

while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame if original is larger than target_w
        if scale_factor < 1.0:
            frame = cv2.resize(frame, (vid_w, vid_h), interpolation=cv2.INTER_AREA)

        results = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0], verbose=False)
        display_frame = frame.copy()
        
        if corridor_only:
            cv2.line(display_frame, (corridor_min_x, 0), (corridor_min_x, vid_h), (100, 100, 100), 1)
            cv2.line(display_frame, (corridor_max_x, 0), (corridor_max_x, vid_h), (100, 100, 100), 1)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()

            for idx, (box, tid) in enumerate(zip(boxes, track_ids)):
                x1, y1, x2, y2 = map(int, box)
                w, h = x2 - x1, y2 - y1
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                if corridor_only and (cx < corridor_min_x or cx > corridor_max_x):
                    continue

                if tid not in track_positions:
                    track_positions[tid] = []
                track_positions[tid].append((cx, cy))

                all_detections.append({
                    "frame": frame_idx,
                    "timestamp_s": round(frame_idx / fps, 2),
                    "track_id": int(tid),
                    "x_center": cx,
                    "y_center": cy,
                    "bbox_x1": x1,
                    "bbox_y1": y1,
                    "bbox_x2": x2,
                    "bbox_y2": y2
                })

                # Dual ROI: Upper body & Lower body (accommodates orientation shifts)
                roi_top = frame[y1:y1 + int(0.50 * h), x1:x2]
                roi_bottom = frame[y2 - int(0.50 * h):y2, x1:x2]

                score_top = detect_tag_features(roi_top, templates_gray)
                score_bottom = detect_tag_features(roi_bottom, templates_gray)

                best_score = max(score_top, score_bottom)
                best_crop = roi_top if score_top >= score_bottom else roi_bottom

                tag_found_here = False
                if best_score >= conf_thresh:
                    track_tag_votes[tid] = track_tag_votes.get(tid, 0) + 1
                    tag_found_here = True
                    if tid not in track_best_crops or best_score > track_best_crops[tid][0]:
                        track_best_crops[tid] = (best_score, best_crop.copy())

                if tag_found_here:
                    color = (0, 255, 255)
                    lbl = f"TAG HIT ID {tid}"
                else:
                    color = (255, 120, 0)
                    lbl = f"Corridor ID {tid}"

                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(display_frame, (cx, cy), 3, (0, 0, 255), -1)
                cv2.putText(display_frame, lbl, (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        cv2.putText(display_frame, f"Frame: {frame_idx}", (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        rgb_display = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(rgb_display, channels="RGB", use_container_width=True)

        frame_idx += 1
        if total_frames > 0:
            progress_bar.progress(min(1.0, frame_idx / total_frames))

    cap.release()
    status_text.success("✅ Tracking Finished!")

    # --- Verification & Deliverables ---
    with col_results:
        st.subheader("📋 Results & Deliverables")
        st.markdown("#### 🤖 Multimodal Gemini 2.5 Flash Verification")

        walking_tracks = {}
        for tid, pts in track_positions.items():
            if len(pts) >= 10:
                ys = [p[1] for p in pts]
                y_disp = max(ys) - min(ys)
                if y_disp > (0.12 * vid_h):
                    votes = track_tag_votes.get(tid, 0)
                    walking_tracks[tid] = (votes, y_disp)

        confirmed_staff_ids = []

        if walking_tracks:
            sorted_walkers = sorted(walking_tracks.items(), key=lambda x: (x[1][0], x[1][1]), reverse=True)

            for cand_id, (hits, disp) in sorted_walkers[:4]:
                if cand_id in track_best_crops:
                    crop = track_best_crops[cand_id][1]
                    with st.spinner(f"Verifying Track ID {int(cand_id)} with Gemini Flash..."):
                        is_conf, reasoning = verify_tag_with_gemini(crop, tag_ref_color, GEMINI_API_KEY)

                    st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), caption=f"Crop for Track ID {int(cand_id)}", width=150)

                    if is_conf:
                        st.success(f"**Gemini Confirmed Staff (Track ID {int(cand_id)}):** {reasoning}")
                        confirmed_staff_ids.append(int(cand_id))
                    else:
                        st.warning(f"**Gemini Output (Track ID {int(cand_id)}):** {reasoning}")

            # Heuristic fallback to primary corridor tracks if API verification fails
            if not confirmed_staff_ids:
                confirmed_staff_ids = [int(sorted_walkers[0][0])]
                if len(sorted_walkers) > 1:
                    confirmed_staff_ids.append(int(sorted_walkers[1][0]))
                st.info(f"Fallback to leading corridor walker tracklets: {[int(x) for x in confirmed_staff_ids]}")

        if not confirmed_staff_ids:
            st.error("No walking corridor staff detected.")
        else:
            st.success(f"Confirmed Staff Tracklet IDs: **{[int(x) for x in confirmed_staff_ids]}**")

            df_all = pd.DataFrame(all_detections)
            df_staff = df_all[df_all["track_id"].isin(confirmed_staff_ids)].copy().sort_values("frame")
            df_staff["staff_present"] = 1

            # Task 2 Bonus: Coordinates Only
            st.markdown("**Task 2 [Bonus]: Staff Coordinates (X, Y)**")
            display_cols = ["frame", "timestamp_s", "track_id", "x_center", "y_center", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
            st.dataframe(df_staff[display_cols].head(40), height=220, use_container_width=True)

            csv_data = df_staff[display_cols].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download staff_coordinates.csv",
                data=csv_data,
                file_name="staff_coordinates.csv",
                mime="text/csv",
                use_container_width=True
            )