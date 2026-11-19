"""
Faceprint: Robust Face Recognition Under Real-World Conditions
Milestone 6: Complete Interactive Face Recognition Web Application (Gradio)
"""

import os
import sys
import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import gradio as gr

# Import our existing, tested Faceprint modules
from modules.detection import FaceDetector
from modules.feature_extraction import FaceFeatureExtractor
from modules.face_database import FaceDatabase
from modules.matching import FaceMatcher

# ---------------------------------------------------------------------------
# Global Singleton Initializations (Loaded once into memory at startup)
# ---------------------------------------------------------------------------
MODEL_DETECTION_PATH = os.path.join("models", "face_detection_yunet.onnx")
MODEL_RECOGNITION_PATH = os.path.join("models", "face_recognition_sface.onnx")
DATABASE_PATH = os.path.join("database", "face_database.npz")

print("[Faceprint] Initializing deep learning models and database...")
detector = FaceDetector(model_path=MODEL_DETECTION_PATH, score_threshold=0.5)
extractor = FaceFeatureExtractor(model_path=MODEL_RECOGNITION_PATH)
database = FaceDatabase(db_path=DATABASE_PATH, auto_load=True)
matcher = FaceMatcher(
    database=database,
    detector=detector,
    extractor=extractor,
    threshold=0.363,
    aggregation="hybrid",
)

# In-memory activity history log
activity_history: List[Dict[str, Any]] = [
    {
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Module": "System Initialized",
        "Result": f"Gallery loaded with {len(database)} identities",
        "Similarity": "-",
        "Status": "Ready",
    }
]


def log_activity(module_name: str, result_name: str, similarity: Union[float, str], status: str):
    """Logs an activity event into the in-memory activity journal."""
    sim_display = f"{similarity:.4f}" if isinstance(similarity, (int, float)) else str(similarity)
    record = {
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Module": module_name,
        "Result": result_name,
        "Similarity": sim_display,
        "Status": status,
    }
    activity_history.insert(0, record)
    # Keep only the latest 30 events
    if len(activity_history) > 30:
        activity_history.pop()


# ---------------------------------------------------------------------------
# Helper UI Rendering & Image Conversion Utilities
# ---------------------------------------------------------------------------
def _gradio_to_bgr(image: np.ndarray) -> np.ndarray:
    """Safely converts RGB, RGBA, or grayscale images from Gradio to OpenCV BGR."""
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def get_dashboard_stats():
    """Returns updated stats for the Dashboard."""
    num_identities = len(database)
    num_templates = database.count_total_embeddings()
    yunet_ok = os.path.isfile(MODEL_DETECTION_PATH)
    sface_ok = os.path.isfile(MODEL_RECOGNITION_PATH)

    identities_list = database.get_identities()
    identities_str = ", ".join(identities_list) if identities_list else "None enrolled yet"

    html_stats = f"""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 1.5rem;">
        <div class="stat-card">
            <div style="font-size: 0.85rem; color: #94A3B8; font-weight: 500;">ENROLLED IDENTITIES</div>
            <div style="font-size: 2.2rem; font-weight: 700; color: #38BDF8; margin-top: 0.25rem;">{num_identities}</div>
            <div style="font-size: 0.75rem; color: #64748B;">Registered persons in local gallery</div>
        </div>
        <div class="stat-card">
            <div style="font-size: 0.85rem; color: #94A3B8; font-weight: 500;">STORED EMBEDDINGS</div>
            <div style="font-size: 2.2rem; font-weight: 700; color: #A855F7; margin-top: 0.25rem;">{num_templates}</div>
            <div style="font-size: 0.75rem; color: #64748B;">128-D facial feature vectors</div>
        </div>
        <div class="stat-card">
            <div style="font-size: 0.85rem; color: #94A3B8; font-weight: 500;">YUNET DETECTOR</div>
            <div style="font-size: 1.4rem; font-weight: 600; color: {'#10B981' if yunet_ok else '#EF4444'}; margin-top: 0.5rem;">
                {'● Active & Loaded' if yunet_ok else '○ Offline'}
            </div>
            <div style="font-size: 0.75rem; color: #64748B;">OpenCV DNN ONNX (320x320)</div>
        </div>
        <div class="stat-card">
            <div style="font-size: 0.85rem; color: #94A3B8; font-weight: 500;">SFACE RECOGNIZER</div>
            <div style="font-size: 1.4rem; font-weight: 600; color: {'#10B981' if sface_ok else '#EF4444'}; margin-top: 0.5rem;">
                {'● Active & Loaded' if sface_ok else '○ Offline'}
            </div>
            <div style="font-size: 0.75rem; color: #64748B;">128-D MobileFaceNet Engine</div>
        </div>
    </div>
    <div style="background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 1rem; margin-bottom: 1rem;">
        <div style="font-size: 0.9rem; font-weight: 600; color: #E2E8F0; margin-bottom: 0.25rem;">Enrolled Gallery Subjects:</div>
        <div style="font-size: 0.85rem; color: #94A3B8; font-family: monospace;">{identities_str}</div>
    </div>
    """
    history_rows = [
        [h["Timestamp"], h["Module"], h["Result"], h["Similarity"], h["Status"]]
        for h in activity_history
    ]
    return html_stats, history_rows


# ---------------------------------------------------------------------------
# Module 1: Registration Logic
# ---------------------------------------------------------------------------
def register_face_action(
    name: str,
    image_rgb: Optional[np.ndarray],
    allow_append: bool,
    consent_confirmed: bool,
    multi_files: Optional[List[Any]] = None,
):
    """Handles enrolling a new face (single or multiple photos) or appending to an existing identity."""
    if not consent_confirmed:
        return (
            """<div class="banner-box error-box">
                <b>⚠️ Informed Consent Required:</b> You must confirm that the individual has consented to biometric template enrollment before registering.
               </div>""",
            None,
            "Registration halted: Missing consent confirmation.",
            *get_dashboard_stats(),
        )

    if not name or not name.strip():
        return (
            """<div class="banner-box error-box">
                <b>⚠️ Invalid Name:</b> Please enter a valid person's name or identity identifier.
               </div>""",
            None,
            "Registration halted: Empty identity name.",
            *get_dashboard_stats(),
        )

    has_multi = bool(multi_files and len(multi_files) > 0)
    if image_rgb is None and not has_multi:
        return (
            """<div class="banner-box error-box">
                <b>⚠️ No Photograph Provided:</b> Please upload an image, capture a webcam frame, or select multiple photographs.
               </div>""",
            None,
            "Registration halted: No image provided.",
            *get_dashboard_stats(),
        )

    clean_name = name.strip()

    # Check for existing identity BEFORE processing to prevent silent overwrite
    already_exists = clean_name in database
    if already_exists and not allow_append:
        return (
            f"""<div class="banner-box warning-box">
                <b>⚠️ Identity Already Registered:</b> <code>'{clean_name}'</code> is already in the database with {len(database.get_embeddings(clean_name))} photo(s).<br>
                To add these new photograph(s) as additional templates, check <i>'Add to existing identity'</i> and click Register again.
               </div>""",
            None,
            f"Duplicate blocked: '{clean_name}' already exists. Silent overwrite prevented.",
            *get_dashboard_stats(),
        )

    # Gather all input images (single upload/webcam + optional multi-file batch)
    images_to_process: List[Tuple[str, np.ndarray]] = []
    skipped_notes: List[str] = []

    if image_rgb is not None:
        images_to_process.append(("Single/Webcam Photo", _gradio_to_bgr(image_rgb)))

    if has_multi:
        for f_obj in multi_files:  # type: ignore
            fpath = f_obj.name if hasattr(f_obj, "name") else str(f_obj)
            fname = os.path.basename(fpath)
            loaded_bgr = cv2.imread(fpath)
            if loaded_bgr is None:
                skipped_notes.append(f"{fname} (could not decode image)")
            else:
                images_to_process.append((fname, loaded_bgr))

    extracted_embeddings: List[np.ndarray] = []
    per_photo_logs: List[str] = []
    preview_annotated_rgb: Optional[np.ndarray] = None
    first_conf = 0.0
    multi_face_note = ""

    for label, bgr_img in images_to_process:
        faces = detector.detect_faces(bgr_img)
        if faces is None or len(faces) == 0:
            skipped_notes.append(f"{label} (no face detected)")
            continue

        primary_face = faces[0]
        conf = float(primary_face[14])

        try:
            aligned_crop = extractor.align_crop(bgr_img, primary_face)
            embedding = extractor.extract_feature(aligned_crop)
            extracted_embeddings.append(embedding)
            per_photo_logs.append(f"{label} (det {conf * 100:.1f}%)")

            if preview_annotated_rgb is None:
                annotated_bgr = detector.draw_faces(bgr_img, faces)
                preview_annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                first_conf = conf
                if len(faces) > 1:
                    multi_face_note = f"<br>• <i>Note: {len(faces)} faces detected in preview image; enrolled highest-confidence face.</i>"
        except Exception as e:
            skipped_notes.append(f"{label} (extraction error: {e})")

    if not extracted_embeddings:
        details = "; ".join(skipped_notes) if skipped_notes else "0 faces detected."
        return (
            f"""<div class="banner-box error-box">
                <b>❌ Face Detection / Extraction Failed:</b> Could not extract valid facial embeddings.<br>
                <span style="font-size: 0.85rem; color: #CBD5E1;">Details: {details}</span>
               </div>""",
            None,
            f"Registration failed: {details}",
            *get_dashboard_stats(),
        )

    # Register all valid embeddings into FaceDatabase
    try:
        new_count = database.register_face(clean_name, extracted_embeddings, allow_append=allow_append)
        database.save()  # Persist to disk
        matcher.database = database  # Keep matcher synced
    except Exception as e:
        return (
            f"""<div class="banner-box error-box">
                <b>❌ Registration Failed:</b> {e}
               </div>""",
            None,
            f"Database error: {e}",
            *get_dashboard_stats(),
        )

    log_activity(
        module_name="Registration",
        result_name=clean_name,
        similarity="N/A",
        status=f"Enrolled ({new_count} total template{'s' if new_count > 1 else ''})",
    )

    action_label = "Updated existing identity" if already_exists else "Enrolled new identity"
    skipped_html = (
        f"<br>• Skipped: <span style='color: #FDE047;'>{', '.join(skipped_notes)}</span>"
        if skipped_notes
        else ""
    )
    success_html = f"""
    <div class="banner-box success-box">
        <div style="font-size: 1.1rem; font-weight: 700; margin-bottom: 0.35rem;">✓ {action_label} Successfully!</div>
        <div>Registered <b>{len(extracted_embeddings)}</b> new photo(s) for <b>{clean_name}</b> into local Faceprint gallery.</div>
        <div style="margin-top: 0.5rem; font-size: 0.85rem; color: #CBD5E1;">
            • Processed photos: <b>{', '.join(per_photo_logs)}</b><br>
            • Total enrolled templates for this subject: <b>{new_count}</b><br>
            • Preprocessing & vectors: <b>Controlled low-light Gamma+CLAHE &amp; unit-normalized 128-D embeddings</b><br>
            • Database status: <b>Saved persistently to disk</b>{skipped_html}{multi_face_note}
        </div>
    </div>
    """

    tech_notes = (
        f"✓ Name: **{clean_name}** | Primary Det Conf: **{first_conf * 100:.1f}%** | "
        f"Enrolled: **{len(extracted_embeddings)}** photo(s) (Normalized 128-D) | Gallery Total: **{len(database)}**"
    )

    return success_html, preview_annotated_rgb, tech_notes, *get_dashboard_stats()


# ---------------------------------------------------------------------------
# Module 2: 1:N Face Recognition Logic
# ---------------------------------------------------------------------------
def recognize_face_action(probe_rgb: Optional[np.ndarray], threshold: float, det_threshold: float = 0.5):
    """Performs 1:N identification on uploaded image or webcam frame."""
    if probe_rgb is None:
        return (
            None,
            """<div class="banner-box error-box"><b>⚠️ Please upload or capture an image to identify.</b></div>""",
            [],
            *get_dashboard_stats(),
        )

    if len(database) == 0:
        return (
            None,
            """<div class="banner-box warning-box"><b>⚠️ Gallery is Empty:</b> No identities are registered yet. Please register at least one face before running identification.</div>""",
            [],
            *get_dashboard_stats(),
        )

    bgr_img = _gradio_to_bgr(probe_rgb)
    annotated_bgr = bgr_img.copy()

    # Detect faces with configurable YuNet detection score threshold
    faces = detector.detect_faces(bgr_img, score_threshold=det_threshold)
    if faces is None or len(faces) == 0:
        log_activity("Identification", "None", 0.0, "No face detected")
        return (
            probe_rgb,
            f"""<div class="banner-box error-box">
                <div style="font-size: 1.1rem; font-weight: 700; margin-bottom: 0.25rem;">❌ No Face Detected</div>
                <div>YuNet could not detect any faces above confidence <b>{det_threshold:.2f}</b>.</div>
                <div style="margin-top: 0.4rem; font-size: 0.85rem; color: #CBD5E1;">
                    • If the person is wearing a mask, dark glasses, or in dim light, try lowering the <i>Face Detection Threshold</i> slider slightly (e.g. 0.40) or improving lighting.
                </div>
               </div>""",
            [],
            *get_dashboard_stats(),
        )

    # Process all detected faces (supports multiple people in probe frame)
    num_faces = len(faces)
    face_summaries = []
    top_candidates_table = []
    primary_name = "Unknown"
    primary_sim = 0.0
    primary_det_conf = float(faces[0][14])
    primary_status = "Processing error"

    for i, face in enumerate(faces):
        x, y, w, h = face[0:4].astype(int)
        x = max(0, x)
        y = max(0, y)
        det_conf = float(face[14])

        try:
            aligned = extractor.align_crop(bgr_img, face)
            emb = extractor.extract_feature(aligned)
            id_res = matcher.identify(emb, threshold=threshold, top_k=5)
            predicted_name = id_res["identity"]
            sim = float(id_res["similarity"])
            status = id_res["status"]

            if i == 0:
                # Store candidate ranking for primary face
                top_candidates_table = [
                    [c["identity"], f"{c['similarity']:.4f}", "Match" if c["similarity"] >= threshold else "Below Threshold"]
                    for c in id_res["candidates"]
                ]
                primary_name = predicted_name
                primary_sim = sim
                primary_det_conf = det_conf
                primary_status = status

            # Color coding: Green for Recognized, Amber/Orange for Unknown
            is_recognized = predicted_name != "Unknown"
            box_color = (0, 255, 0) if is_recognized else (0, 165, 255)
            badge_text = f"{predicted_name} ({sim:.2f})"

            # Draw bounding box
            cv2.rectangle(annotated_bgr, (x, y), (x + w, y + h), box_color, 2)

            # Draw text badge
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            font_thickness = 1
            (tw, th), baseline = cv2.getTextSize(badge_text, font, font_scale, font_thickness)
            label_y = y - 8 if y - th - 8 > 0 else y + th + 8
            cv2.rectangle(annotated_bgr, (x, label_y - th - 2), (x + tw + 4, label_y + baseline), box_color, cv2.FILLED)
            cv2.putText(annotated_bgr, badge_text, (x + 2, label_y), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

            face_summaries.append(f"Face #{i+1}: <b>{predicted_name}</b> (Det: {det_conf*100:.1f}%, Sim: {sim:.4f})")
        except Exception as e:
            face_summaries.append(f"Face #{i+1} processing error: {e}")

    # Log primary face activity
    log_activity(
        module_name="Identification",
        result_name=primary_name,
        similarity=primary_sim,
        status="Recognized" if primary_name != "Unknown" else "Unknown Person",
    )

    # Build output card
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    multi_summary_html = (
        "<div style='margin-top: 0.5rem; font-size: 0.8rem; color: #CBD5E1; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 0.4rem;'>"
        + "<br>".join(face_summaries)
        + "</div>"
        if num_faces > 1
        else ""
    )

    if primary_name != "Unknown":
        result_html = f"""
        <div class="banner-box success-box">
            <div style="font-size: 0.85rem; color: #86EFAC; font-weight: 600; text-transform: uppercase;">Face Detected & Recognized (1:N)</div>
            <div style="font-size: 1.8rem; font-weight: 800; color: #F8FAFC; margin: 0.25rem 0;">{primary_name}</div>
            <div style="display: flex; gap: 0.75rem; align-items: center; margin-top: 0.5rem; flex-wrap: wrap;">
                <span class="badge-match">● RECOGNIZED</span>
                <span style="font-size: 0.9rem; color: #E2E8F0;">Cosine Similarity: <b>{primary_sim:.4f}</b></span>
                <span style="font-size: 0.8rem; color: #94A3B8;">(Match Threshold: {threshold:.3f})</span>
            </div>
            <div style="margin-top: 0.5rem; font-size: 0.8rem; color: #CBD5E1;">
                Face detection succeeded ({num_faces} face(s), YuNet score: {primary_det_conf:.2f}). Vector matches enrolled 128-D template.
            </div>
            {multi_summary_html}
        </div>
        """
    else:
        result_html = f"""
        <div class="banner-box unknown-box">
            <div style="font-size: 0.85rem; color: #FDE047; font-weight: 600; text-transform: uppercase;">Face Detected — Unknown Identity</div>
            <div style="font-size: 1.8rem; font-weight: 800; color: #FEF08A; margin: 0.25rem 0;">UNKNOWN PERSON</div>
            <div style="display: flex; gap: 0.75rem; align-items: center; margin-top: 0.5rem; flex-wrap: wrap;">
                <span class="badge-unknown">● BELOW THRESHOLD</span>
                <span style="font-size: 0.9rem; color: #FEF08A;">Top Similarity: <b>{primary_sim:.4f}</b></span>
                <span style="font-size: 0.8rem; color: #CA8A04;">(Match Threshold: {threshold:.3f})</span>
            </div>
            <div style="margin-top: 0.5rem; font-size: 0.8rem; color: #FEF9C3;">
                Face detection succeeded ({num_faces} face(s), YuNet score: {primary_det_conf:.2f}), but no enrolled identity met the similarity threshold. Note: Heavy occlusion (such as masks) reduces facial embedding similarity.
            </div>
            {multi_summary_html}
        </div>
        """

    return annotated_rgb, result_html, top_candidates_table, *get_dashboard_stats()


# ---------------------------------------------------------------------------
# Module 3: 1:1 Face Verification Logic
# ---------------------------------------------------------------------------
def verify_faces_action(image_a: Optional[np.ndarray], image_b: Optional[np.ndarray], threshold: float):
    """Performs 1:1 verification between two distinct photographs."""
    if image_a is None or image_b is None:
        return (
            None,
            None,
            """<div class="banner-box error-box"><b>⚠️ Please upload both Photograph A and Photograph B to verify.</b></div>""",
            *get_dashboard_stats(),
        )

    bgr_a = _gradio_to_bgr(image_a)
    bgr_b = _gradio_to_bgr(image_b)

    faces_a = detector.detect_faces(bgr_a)
    faces_b = detector.detect_faces(bgr_b)

    if faces_a is None or len(faces_a) == 0:
        return (
            image_a,
            image_b,
            """<div class="banner-box error-box"><b>❌ Photo A Error:</b> No face detected in Photograph A.</div>""",
            *get_dashboard_stats(),
        )

    if faces_b is None or len(faces_b) == 0:
        return (
            image_a,
            image_b,
            """<div class="banner-box error-box"><b>❌ Photo B Error:</b> No face detected in Photograph B.</div>""",
            *get_dashboard_stats(),
        )

    multi_notes = []
    if len(faces_a) > 1:
        multi_notes.append(f"Photo A contained {len(faces_a)} faces (used primary face).")
    if len(faces_b) > 1:
        multi_notes.append(f"Photo B contained {len(faces_b)} faces (used primary face).")
    multi_note_html = (
        f"<div style='margin-top: 0.35rem; font-size: 0.8rem; color: #FDE047;'>⚠️ {' '.join(multi_notes)}</div>"
        if multi_notes
        else ""
    )

    # Annotate faces
    ann_a = detector.draw_faces(bgr_a, faces_a)
    ann_b = detector.draw_faces(bgr_b, faces_b)
    ann_a_rgb = cv2.cvtColor(ann_a, cv2.COLOR_BGR2RGB)
    ann_b_rgb = cv2.cvtColor(ann_b, cv2.COLOR_BGR2RGB)

    try:
        aligned_a = extractor.align_crop(bgr_a, faces_a[0])
        aligned_b = extractor.align_crop(bgr_b, faces_b[0])

        emb_a = extractor.extract_feature(aligned_a)
        emb_b = extractor.extract_feature(aligned_b)

        similarity = matcher.compute_cosine_similarity(emb_a, emb_b)
        is_match = similarity >= threshold
    except Exception as e:
        return (
            ann_a_rgb,
            ann_b_rgb,
            f"""<div class="banner-box error-box"><b>❌ Verification Processing Error:</b> {e}</div>""",
            *get_dashboard_stats(),
        )

    log_activity(
        module_name="1:1 Verification",
        result_name="Photo A vs Photo B",
        similarity=similarity,
        status="Match" if is_match else "Mismatch",
    )

    if is_match:
        result_html = f"""
        <div class="banner-box success-box">
            <div style="font-size: 1.8rem; font-weight: 800; color: #34D399; margin-bottom: 0.25rem;">MATCH (SAME PERSON)</div>
            <div style="font-size: 1rem; color: #E2E8F0;">
                Calculated Cosine Similarity: <b style="font-size: 1.2rem; color: #38BDF8;">{similarity:.4f}</b>
                <span style="font-size: 0.85rem; color: #94A3B8;">(Threshold: {threshold:.3f})</span>
            </div>
            <div style="margin-top: 0.5rem; font-size: 0.85rem; color: #CBD5E1;">
                ✓ The 128-dimensional biometric embeddings confirm that Photograph A and Photograph B belong to the same identity.
            </div>
            {multi_note_html}
        </div>
        """
    else:
        result_html = f"""
        <div class="banner-box mismatch-box">
            <div style="font-size: 1.8rem; font-weight: 800; color: #F87171; margin-bottom: 0.25rem;">NO MATCH (DIFFERENT PEOPLE)</div>
            <div style="font-size: 1rem; color: #E2E8F0;">
                Calculated Cosine Similarity: <b style="font-size: 1.2rem; color: #F87171;">{similarity:.4f}</b>
                <span style="font-size: 0.85rem; color: #94A3B8;">(Threshold: {threshold:.3f})</span>
            </div>
            <div style="margin-top: 0.5rem; font-size: 0.85rem; color: #CBD5E1;">
                ✗ The similarity score falls below the required threshold. The two photographs represent distinct individuals.
            </div>
            {multi_note_html}
        </div>
        """

    return ann_a_rgb, ann_b_rgb, result_html, *get_dashboard_stats()


# ---------------------------------------------------------------------------
# Module 4: Gallery Management Logic
# ---------------------------------------------------------------------------
def get_gallery_table_data():
    """Builds the dataframe rows for the Registered People tab."""
    identities = database.get_identities()
    rows = []
    for name in identities:
        embs = database.get_embeddings(name)
        rows.append([name, embs.shape[0], f"{embs.shape[1]}-D SFace", "Persistent (.npz)"])
    return rows


def delete_identity_action(selected_name: str, confirm_checkbox: bool):
    """Deletes an identity from the local database."""
    if not selected_name:
        return "⚠️ Please select an identity to delete.", get_gallery_table_data(), gr.update(), *get_dashboard_stats()

    if not confirm_checkbox:
        return (
            f"⚠️ Deletion halted: Please check 'I confirm deletion of {selected_name}' before clicking Delete.",
            get_gallery_table_data(),
            gr.update(),
            *get_dashboard_stats(),
        )

    deleted = database.remove_identity(selected_name)
    if deleted:
        database.save()
        matcher.database = database
        log_activity("Gallery Manager", selected_name, "N/A", "Deleted")
        identities = database.get_identities()
        return (
            f"✓ Successfully deleted '{selected_name}' and all associated templates from the database.",
            get_gallery_table_data(),
            gr.update(choices=identities, value=identities[0] if identities else None),
            *get_dashboard_stats(),
        )
    else:
        return f"⚠️ Identity '{selected_name}' not found in database.", get_gallery_table_data(), gr.update(), *get_dashboard_stats()


def add_photos_to_identity_action(
    selected_name: str,
    image_rgb: Optional[np.ndarray],
    multi_files: Optional[List[Any]],
    consent_confirmed: bool,
):
    """Appends one or more additional photographs (e.g. low-light, spectacles, angles) to a registered identity."""
    if not selected_name:
        return "⚠️ Please select an existing identity.", get_gallery_table_data(), *get_dashboard_stats()

    if not consent_confirmed:
        return "⚠️ Informed consent confirmation is required to enroll additional photographs.", get_gallery_table_data(), *get_dashboard_stats()

    has_multi = bool(multi_files and len(multi_files) > 0)
    if image_rgb is None and not has_multi:
        return "⚠️ Please upload at least one photograph or webcam snapshot to add.", get_gallery_table_data(), *get_dashboard_stats()

    images_to_process: List[Tuple[str, np.ndarray]] = []
    skipped_notes: List[str] = []

    if image_rgb is not None:
        images_to_process.append(("Uploaded/Webcam Photo", _gradio_to_bgr(image_rgb)))
    if has_multi:
        for f_obj in multi_files:  # type: ignore
            fpath = f_obj.name if hasattr(f_obj, "name") else str(f_obj)
            fname = os.path.basename(fpath)
            loaded = cv2.imread(fpath)
            if loaded is None:
                skipped_notes.append(f"{fname} (unreadable)")
            else:
                images_to_process.append((fname, loaded))

    added_embeddings: List[np.ndarray] = []
    for label, bgr_img in images_to_process:
        faces = detector.detect_faces(bgr_img)
        if faces is None or len(faces) == 0:
            skipped_notes.append(f"{label} (no face detected)")
            continue
        try:
            aligned = extractor.align_crop(bgr_img, faces[0])
            emb = extractor.extract_feature(aligned)
            added_embeddings.append(emb)
        except Exception as e:
            skipped_notes.append(f"{label} (error: {e})")

    if not added_embeddings:
        return (
            f"❌ Failed to add photographs to '{selected_name}': {'; '.join(skipped_notes)}",
            get_gallery_table_data(),
            *get_dashboard_stats(),
        )

    total_count = database.register_face(selected_name, added_embeddings, allow_append=True)
    database.save()
    matcher.database = database
    log_activity("Gallery Manager", selected_name, "N/A", f"Added {len(added_embeddings)} template(s)")

    skip_msg = f" (Skipped: {', '.join(skipped_notes)})" if skipped_notes else ""
    return (
        f"✓ Added {len(added_embeddings)} normalized template(s) to '{selected_name}' (Total templates: {total_count}).{skip_msg}",
        get_gallery_table_data(),
        *get_dashboard_stats(),
    )


def load_demo_dataset_action():
    """Loads baseline synthetic dataset (Alex, Sam, Taylor) into gallery for immediate testing."""
    test_data_dir = "Faceprint_Test_Dataset"
    if os.path.isdir(os.path.join(test_data_dir, "faceprint_synthetic_dataset")):
        test_data_dir = os.path.join(test_data_dir, "faceprint_synthetic_dataset")

    if not os.path.isdir(test_data_dir):
        return f"❌ Dataset directory not found at '{test_data_dir}'.", get_gallery_table_data(), gr.update(), *get_dashboard_stats()

    enrolled_count = 0
    for subject in sorted(os.listdir(test_data_dir)):
        sub_path = os.path.join(test_data_dir, subject)
        if os.path.isdir(sub_path):
            orig_img = os.path.join(sub_path, "original.png")
            if os.path.isfile(orig_img):
                img = cv2.imread(orig_img)
                if img is not None:
                    faces = detector.detect_faces(img)
                    if faces is not None and len(faces) > 0:
                        aligned = extractor.align_crop(img, faces[0])
                        emb = extractor.extract_feature(aligned)
                        database.register_face(subject, emb, allow_append=True)
                        enrolled_count += 1

    database.save()
    matcher.database = database
    log_activity("Gallery Manager", "Demo Dataset", "N/A", f"Imported {enrolled_count} subjects")
    identities = database.get_identities()
    return (
        f"✓ Successfully imported {enrolled_count} demo subjects from Faceprint_Test_Dataset!",
        get_gallery_table_data(),
        gr.update(choices=identities, value=identities[0] if identities else None),
        *get_dashboard_stats(),
    )


def clear_history_action():
    """Clears the in-memory recognition activity log."""
    global activity_history
    activity_history = [
        {
            "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Module": "History Cleared",
            "Result": "Activity log reset by user",
            "Similarity": "-",
            "Status": "Ready",
        }
    ]
    return get_dashboard_stats()[1]


# ---------------------------------------------------------------------------
# Custom CSS Styling (Dark, Professional, High-Tech Theme)
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
:root {
    --bg-dark: #0B0F19;
    --card-dark: #1E293B;
    --accent-blue: #38BDF8;
    --accent-purple: #818CF8;
    --accent-green: #34D399;
}

body, .gradio-container {
    background-color: #0B0F19 !important;
    color: #F8FAFC !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}

.main-header {
    text-align: center;
    padding: 1.75rem 1rem;
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
    border-radius: 16px;
    border: 1px solid rgba(56, 189, 248, 0.25);
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
}

.stat-card {
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 1.25rem 1rem;
    text-align: center;
}

.banner-box {
    padding: 1.25rem;
    border-radius: 12px;
    margin: 1rem 0;
    font-size: 0.95rem;
    line-height: 1.5;
}

.success-box {
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.4);
    color: #F8FAFC;
}

.warning-box {
    background: rgba(245, 158, 11, 0.12);
    border: 1px solid rgba(245, 158, 11, 0.4);
    color: #F8FAFC;
}

.error-box {
    background: rgba(239, 68, 68, 0.12);
    border: 1px solid rgba(239, 68, 68, 0.4);
    color: #F8FAFC;
}

.mismatch-box {
    background: rgba(239, 68, 68, 0.15);
    border: 1px solid rgba(239, 68, 68, 0.5);
    color: #F8FAFC;
}

.unknown-box {
    background: rgba(234, 179, 8, 0.12);
    border: 1px solid rgba(234, 179, 8, 0.5);
    color: #F8FAFC;
}

.badge-match {
    background: rgba(16, 185, 129, 0.2);
    border: 1px solid #10B981;
    color: #34D399;
    padding: 4px 12px;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 0.8rem;
    display: inline-block;
}

.badge-unknown {
    background: rgba(234, 179, 8, 0.2);
    border: 1px solid #EAB308;
    color: #FACC15;
    padding: 4px 12px;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 0.8rem;
    display: inline-block;
}

button.primary-btn {
    background: linear-gradient(135deg, #2563EB 0%, #4F46E5 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
}
"""

# ---------------------------------------------------------------------------
# Gradio Application Definition
# ---------------------------------------------------------------------------
def create_app() -> gr.Blocks:
    with gr.Blocks(title="FACEPRINT - AI Face Recognition") as app:

        # Header Title Banner
        gr.HTML("""
        <div class="main-header">
            <div style="font-size: 0.85rem; letter-spacing: 0.2em; text-transform: uppercase; color: #38BDF8; font-weight: 700; margin-bottom: 0.25rem;">
                BIOMETRIC VISION PLATFORM
            </div>
            <div style="font-size: 2.5rem; font-weight: 900; letter-spacing: -0.02em; background: linear-gradient(90deg, #FFFFFF 0%, #38BDF8 50%, #818CF8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                FACEPRINT
            </div>
            <div style="font-size: 1rem; color: #94A3B8; margin-top: 0.25rem;">
                Robust Face Recognition System Under Real-World Conditions
            </div>
            <div style="font-size: 0.75rem; color: #64748B; margin-top: 0.5rem;">
                OpenCV YuNet Detection • OpenCV SFace 128-D Embeddings • NumPy Persistent Local Storage
            </div>
        </div>
        """)

        # Main Navigation Tabs
        with gr.Tabs() as main_tabs:

            # ---------------------------------------------------------------
            # TAB 1: SYSTEM DASHBOARD
            # ---------------------------------------------------------------
            with gr.TabItem("📊 System Dashboard", id="tab_dashboard"):
                initial_stats_html, initial_history_rows = get_dashboard_stats()

                dashboard_stats_display = gr.HTML(initial_stats_html)

                gr.HTML("""
                <div style="background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 1.25rem; margin-bottom: 1.5rem;">
                    <div style="font-size: 1.05rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.5rem;">System Architecture & Biometric Pipeline</div>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 0.75rem; text-align: center; margin-top: 0.75rem;">
                        <div style="background: rgba(15, 23, 42, 0.6); padding: 0.75rem; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.05);">
                            <div style="color: #38BDF8; font-weight: 700; font-size: 0.85rem;">1. Image Input</div>
                            <div style="color: #94A3B8; font-size: 0.75rem;">Upload or Webcam capture</div>
                        </div>
                        <div style="background: rgba(15, 23, 42, 0.6); padding: 0.75rem; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.05);">
                            <div style="color: #38BDF8; font-weight: 700; font-size: 0.85rem;">2. Face Detection</div>
                            <div style="color: #94A3B8; font-size: 0.75rem;">YuNet bbox & 5 landmarks</div>
                        </div>
                        <div style="background: rgba(15, 23, 42, 0.6); padding: 0.75rem; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.05);">
                            <div style="color: #818CF8; font-weight: 700; font-size: 0.85rem;">3. Face Alignment</div>
                            <div style="color: #94A3B8; font-size: 0.75rem;">112x112 affine normalization</div>
                        </div>
                        <div style="background: rgba(15, 23, 42, 0.6); padding: 0.75rem; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.05);">
                            <div style="color: #818CF8; font-weight: 700; font-size: 0.85rem;">4. Embedding</div>
                            <div style="color: #94A3B8; font-size: 0.75rem;">SFace 128-D vector</div>
                        </div>
                        <div style="background: rgba(15, 23, 42, 0.6); padding: 0.75rem; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.05);">
                            <div style="color: #34D399; font-weight: 700; font-size: 0.85rem;">5. Matching</div>
                            <div style="color: #94A3B8; font-size: 0.75rem;">Cosine Similarity & threshold</div>
                        </div>
                    </div>
                </div>
                """)

                gr.Markdown("### 🕒 Recent Recognition & Verification Activity")
                activity_table = gr.Dataframe(
                    headers=["Timestamp", "Module", "Result", "Similarity", "Status"],
                    value=initial_history_rows,
                    interactive=False,
                )

                with gr.Row():
                    refresh_dash_btn = gr.Button("🔄 Refresh Dashboard", variant="secondary")
                    clear_history_btn = gr.Button("🗑️ Clear Activity History", variant="secondary")

            # ---------------------------------------------------------------
            # TAB 2: REGISTER NEW FACE
            # ---------------------------------------------------------------
            with gr.TabItem("👤 Register New Face", id="tab_register"):
                gr.Markdown("### Enroll a Person into the Local Faceprint Gallery")
                gr.Markdown(
                    "Upload a clear portrait or take a photo with your webcam. "
                    "YuNet will detect the face and landmarks, SFace will extract a 128-D vector, "
                    "and the biometric template will be saved locally."
                )

                with gr.Row():
                    with gr.Column(scale=5):
                        reg_name_input = gr.Textbox(
                            label="Person's Full Name / Identifier",
                            placeholder="e.g. Alex Rivera",
                            lines=1,
                        )
                        reg_image_input = gr.Image(
                            label="Photograph (Upload or Webcam)",
                            sources=["upload", "webcam"],
                            type="numpy",
                        )
                        reg_multi_files = gr.Files(
                            label="Optional: Batch Upload Multiple Photos (Normal, Low-Light, Spectacles, Angles)",
                            file_types=["image"],
                            file_count="multiple",
                        )
                        reg_allow_append = gr.Checkbox(
                            label="Add to existing identity (Multi-photo enrollment)",
                            value=False,
                            info="Check this if the person is already registered and you want to enroll an additional photo.",
                        )
                        reg_consent = gr.Checkbox(
                            label="I confirm this individual has provided informed consent for biometric template enrollment.",
                            value=True,
                            info="Mandatory privacy compliance requirement.",
                        )
                        register_submit_btn = gr.Button("🚀 Enroll Face into Gallery", variant="primary")

                    with gr.Column(scale=5):
                        reg_status_banner = gr.HTML(
                            """<div class="banner-box" style="background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(255,255,255,0.08); color: #94A3B8;">
                                Awaiting registration submission...
                               </div>"""
                        )
                        reg_annotated_preview = gr.Image(
                            label="Detection & Alignment Preview",
                            interactive=False,
                        )
                        reg_tech_notes = gr.Markdown("")

            # ---------------------------------------------------------------
            # TAB 3: 1:N FACE RECOGNITION
            # ---------------------------------------------------------------
            with gr.TabItem("🔍 1:N Face Recognition", id="tab_recognize"):
                gr.Markdown("### Identify an Unknown Face Against the Enrolled Gallery")
                gr.Markdown(
                    "Upload an image or capture a webcam photo. The system will detect faces, extract embeddings, "
                    "and match them against every registered identity. If no candidate exceeds the threshold, "
                    "the face is safely rejected as **UNKNOWN PERSON**."
                )

                with gr.Row():
                    with gr.Column(scale=5):
                        rec_probe_input = gr.Image(
                            label="Probe Photograph (Upload or Webcam)",
                            sources=["upload", "webcam"],
                            type="numpy",
                        )
                        rec_threshold_slider = gr.Slider(
                            minimum=0.20,
                            maximum=0.60,
                            value=0.363,
                            step=0.01,
                            label="Matching Cosine Similarity Threshold",
                            info="Standard SFace benchmark: 0.363. Higher = stricter security; Lower = higher tolerance.",
                        )
                        rec_det_threshold_slider = gr.Slider(
                            minimum=0.30,
                            maximum=0.90,
                            value=0.50,
                            step=0.05,
                            label="Face Detection Confidence Threshold (YuNet)",
                            info="Default 0.50 detects partially occluded faces (masks/spectacles). Lower if faces in dim light are missed.",
                        )
                        recognize_submit_btn = gr.Button("🔎 Search & Identify Person", variant="primary")

                    with gr.Column(scale=5):
                        rec_result_banner = gr.HTML(
                            """<div class="banner-box" style="background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(255,255,255,0.08); color: #94A3B8;">
                                Upload a probe photo and click Search to identify.
                               </div>"""
                        )
                        rec_annotated_output = gr.Image(
                            label="Recognition Bounding Box Output",
                            interactive=False,
                        )
                        gr.Markdown("#### Ranked Gallery Candidates:")
                        rec_candidates_table = gr.Dataframe(
                            headers=["Candidate Identity", "Cosine Similarity", "Threshold Decision"],
                            interactive=False,
                        )

            # ---------------------------------------------------------------
            # TAB 4: 1:1 FACE VERIFICATION
            # ---------------------------------------------------------------
            with gr.TabItem("⚖️ 1:1 Face Verification", id="tab_verify"):
                gr.Markdown("### 1:1 Direct Face Verification")
                gr.Markdown(
                    "Compare two distinct photographs to determine whether they represent the **same individual**. "
                    "The system aligns both faces and calculates the exact cosine similarity angle between their 128-D vectors."
                )

                with gr.Row():
                    with gr.Column(scale=5):
                        ver_image_a = gr.Image(
                            label="Photograph A",
                            sources=["upload", "webcam"],
                            type="numpy",
                        )
                    with gr.Column(scale=5):
                        ver_image_b = gr.Image(
                            label="Photograph B",
                            sources=["upload", "webcam"],
                            type="numpy",
                        )

                with gr.Row():
                    ver_threshold_slider = gr.Slider(
                        minimum=0.20,
                        maximum=0.60,
                        value=0.363,
                        step=0.01,
                        label="Verification Threshold",
                        info="Matches with similarity >= threshold are classified as the SAME PERSON.",
                    )
                    verify_submit_btn = gr.Button("⚖️ Compare & Verify Faces", variant="primary")

                ver_result_banner = gr.HTML(
                    """<div class="banner-box" style="background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(255,255,255,0.08); color: #94A3B8; text-align: center;">
                        Upload Photo A and Photo B and click Compare.
                       </div>"""
                )

                with gr.Row():
                    ver_ann_a = gr.Image(label="Face A Detected", interactive=False)
                    ver_ann_b = gr.Image(label="Face B Detected", interactive=False)

            # ---------------------------------------------------------------
            # TAB 5: REGISTERED PEOPLE (GALLERY MANAGER)
            # ---------------------------------------------------------------
            with gr.TabItem("📁 Registered Gallery", id="tab_gallery"):
                gr.Markdown("### Local Biometric Gallery Management")
                gr.Markdown(
                    "View enrolled identities, inspect the number of stored templates per person, "
                    "or safely remove identities from the local `.npz` database."
                )

                gallery_table = gr.Dataframe(
                    headers=["Identity Name", "Photos Enrolled", "Feature Dimension", "Storage Format"],
                    value=get_gallery_table_data(),
                    interactive=False,
                )

                with gr.Row():
                    refresh_gallery_btn = gr.Button("🔄 Refresh Gallery Table", variant="secondary")
                    load_demo_btn = gr.Button("📦 Import Demo Dataset (Alex, Sam, Taylor)", variant="secondary")

                gr.Markdown("#### Add Additional Photograph(s) to an Enrolled Identity:")
                gr.Markdown(
                    "Enhance robustness across lighting and conditions by enrolling extra photos "
                    "(e.g. low-light, spectacles, different angles) for an existing person."
                )
                with gr.Row():
                    identities_current = database.get_identities()
                    with gr.Column(scale=4):
                        add_photo_dropdown = gr.Dropdown(
                            label="Select Registered Identity",
                            choices=identities_current,
                            value=identities_current[0] if identities_current else None,
                        )
                        add_photo_consent = gr.Checkbox(
                            label="Confirm informed consent for additional photo enrollment",
                            value=True,
                        )
                        add_photo_btn = gr.Button("➕ Add Photo(s) to Selected Identity", variant="primary")
                        add_photo_status = gr.Markdown("")
                    with gr.Column(scale=4):
                        add_photo_image = gr.Image(
                            label="Single Photo / Webcam",
                            sources=["upload", "webcam"],
                            type="numpy",
                        )
                    with gr.Column(scale=4):
                        add_photo_files = gr.Files(
                            label="Batch Upload Additional Photos",
                            file_types=["image"],
                            file_count="multiple",
                        )

                gr.Markdown("#### Delete an Identity from Gallery:")
                with gr.Row():
                    delete_dropdown = gr.Dropdown(
                        label="Select Identity to Delete",
                        choices=identities_current,
                        value=identities_current[0] if identities_current else None,
                    )
                    delete_confirm = gr.Checkbox(
                        label="I confirm deletion of this identity and all stored templates",
                        value=False,
                    )
                    delete_btn = gr.Button("🗑️ Delete Identity", variant="stop")

                delete_status_msg = gr.Markdown("")

            # ---------------------------------------------------------------
            # TAB 6: ABOUT & SYSTEM ARCHITECTURE
            # ---------------------------------------------------------------
            with gr.TabItem("ℹ️ About System", id="tab_about"):
                gr.Markdown("""
                ### About Faceprint: Robust Face Recognition Under Real-World Conditions

                **Faceprint** is an enterprise-grade biometric computer vision application designed to provide reliable, privacy-preserving face recognition under diverse environmental conditions (harsh lighting, low light, extreme poses, and occlusions).

                ---

                #### Core Deep Learning Technologies:

                1. **Face Detection: OpenCV YuNet (`face_detection_yunet.onnx`)**
                   - Lightweight, high-performance deep convolutional network with configurable threshold (`0.50` default), proportional resizing, and safe contrast/padding fallbacks.
                   - Dynamically detects face bounding boxes and **5 key landmark points** (both eyes, nose tip, and mouth corners) in real-time.

                2. **Controlled Illumination & Feature Extraction: OpenCV SFace (`face_recognition_sface.onnx`)**
                   - Aligns and crops each face to a standardized $112 \\times 112$ patch via affine landmark transformations.
                   - Applies identical controlled adaptive **Gamma Correction + LAB CLAHE** preprocessing for dark/low-light faces while leaving well-lit faces unaltered.
                   - Compresses 37,632 color pixel values into a **unit-L2 normalized 128-dimensional embedding vector**.

                3. **Local Vector Database (`database/face_database.npz`)**
                   - Persistent storage using NumPy's compressed binary archive format (`.npz`).
                   - Supports multi-photo enrollment per subject (normal lighting, low lighting, spectacles, and angles) with unit-normalized vectors.

                4. **Matching Engine (`modules/matching.py`)**
                   - Computes geometric cosine similarity between 128-D vectors:
                     $$\\text{Cosine Similarity} = \\frac{\\mathbf{u} \\cdot \\mathbf{v}}{\\|\\mathbf{u}\\|_2 \\|\\mathbf{v}\\|_2}$$
                   - Uses **Hybrid Consensus Template Aggregation** (`0.70 × best_template + 0.30 × identity_centroid`) and configurable threshold (SFace standard baseline: `0.363`) with strict rejection of unknown faces.

                ---

                #### Biometric Ethics & Privacy Compliance:
                - **100% Local Execution**: All image processing and neural network inference runs entirely on your local machine. No photos or biometric data are transmitted over the internet.
                - **Informed Consent Principle**: Enrollment strictly requires informed consent. Covert tracking and automatic background enrollment are strictly prohibited.
                - **Transparent Scoring**: Similarity is reported as pure vector cosine similarity, never as false calibrated probabilities or misleading certainty guarantees.
                """)

        # -------------------------------------------------------------------
        # Event Callbacks and Interactivity Wiring
        # -------------------------------------------------------------------
        # Dashboard events
        refresh_dash_btn.click(
            fn=get_dashboard_stats,
            inputs=[],
            outputs=[dashboard_stats_display, activity_table],
        )
        clear_history_btn.click(
            fn=clear_history_action,
            inputs=[],
            outputs=[activity_table],
        )

        # Registration events
        register_submit_btn.click(
            fn=register_face_action,
            inputs=[reg_name_input, reg_image_input, reg_allow_append, reg_consent, reg_multi_files],
            outputs=[reg_status_banner, reg_annotated_preview, reg_tech_notes, dashboard_stats_display, activity_table],
        ).then(
            fn=get_gallery_table_data,
            inputs=[],
            outputs=[gallery_table],
        ).then(
            fn=lambda: (gr.update(choices=database.get_identities()), gr.update(choices=database.get_identities())),
            inputs=[],
            outputs=[delete_dropdown, add_photo_dropdown],
        )

        # Recognition events
        recognize_submit_btn.click(
            fn=recognize_face_action,
            inputs=[rec_probe_input, rec_threshold_slider, rec_det_threshold_slider],
            outputs=[rec_annotated_output, rec_result_banner, rec_candidates_table, dashboard_stats_display, activity_table],
        )

        # Verification events
        verify_submit_btn.click(
            fn=verify_faces_action,
            inputs=[ver_image_a, ver_image_b, ver_threshold_slider],
            outputs=[ver_ann_a, ver_ann_b, ver_result_banner, dashboard_stats_display, activity_table],
        )

        # Gallery Manager events
        refresh_gallery_btn.click(
            fn=get_gallery_table_data,
            inputs=[],
            outputs=[gallery_table],
        ).then(
            fn=lambda: (gr.update(choices=database.get_identities()), gr.update(choices=database.get_identities())),
            inputs=[],
            outputs=[delete_dropdown, add_photo_dropdown],
        )

        add_photo_btn.click(
            fn=add_photos_to_identity_action,
            inputs=[add_photo_dropdown, add_photo_image, add_photo_files, add_photo_consent],
            outputs=[add_photo_status, gallery_table, dashboard_stats_display, activity_table],
        )

        load_demo_btn.click(
            fn=load_demo_dataset_action,
            inputs=[],
            outputs=[delete_status_msg, gallery_table, delete_dropdown, dashboard_stats_display, activity_table],
        ).then(
            fn=lambda: gr.update(choices=database.get_identities()),
            inputs=[],
            outputs=[add_photo_dropdown],
        )

        delete_btn.click(
            fn=delete_identity_action,
            inputs=[delete_dropdown, delete_confirm],
            outputs=[delete_status_msg, gallery_table, delete_dropdown, dashboard_stats_display, activity_table],
        ).then(
            fn=lambda: gr.update(choices=database.get_identities()),
            inputs=[],
            outputs=[add_photo_dropdown],
        )

    return app


# ---------------------------------------------------------------------------
# Main Execution Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo = create_app()
    # Launch local web server on port 7860
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        css=CUSTOM_CSS,
    )
