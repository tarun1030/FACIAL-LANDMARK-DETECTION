import cv2
import mediapipe as mp
import numpy as np
import tempfile
import streamlit as st

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Facial Features Mapping
FACIAL_FEATURES = {
    "Left Eye": ([33, 160, 158, 133, 153, 144, 145, 23], (0, 255, 0)),
    "Right Eye": ([263, 387, 385, 362, 373, 380, 374, 253], (0, 255, 0)),
    "Nose": ([1, 2, 5, 195, 197], (0, 165, 255)),
    "Mouth": ([61, 146, 91, 181, 84, 17, 314, 405, 321, 375], (147, 20, 255)),
    "Left Eyebrow": ([70, 63, 105, 66, 107], (0, 215, 255)),
    "Right Eyebrow": ([336, 296, 334, 293, 300], (0, 215, 255)),
    "Face Contour": ([234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 
                      377, 400, 378, 379, 365, 397, 288, 361, 323, 454], (255, 20, 147)),
    "Left Ear": ([234, 93, 132, 58], (255, 140, 0)),
    "Right Ear": ([323, 454, 356, 389], (255, 140, 0)),
}

# Function to Process Frames
def process_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        h, w, _ = frame.shape
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            # Draw Head Outline
            head_outline_idx = [
                10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 
                365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 
                132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 338  
            ]
            head_outline_pts = [(int(landmarks[idx].x * w), int(landmarks[idx].y * h)) for idx in head_outline_idx]
            for i in range(len(head_outline_pts) - 1):
                cv2.line(frame, head_outline_pts[i], head_outline_pts[i + 1], (0, 255, 255), 2)

            # Draw Facial Features
            for feature, (indices, color) in FACIAL_FEATURES.items():
                x_sum, y_sum = 0, 0  # For text placement
                for i in range(len(indices)):
                    idx = indices[i]
                    x, y = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
                    x_sum += x
                    y_sum += y
                    cv2.circle(frame, (x, y), 2, color, -1)
                    if i < len(indices) - 1:
                        idx_next = indices[i + 1]
                        x_next, y_next = int(landmarks[idx_next].x * w), int(landmarks[idx_next].y * h)
                        cv2.line(frame, (x, y), (x_next, y_next), color, 1)

                # Display Label Near Feature
                label_x = x_sum // len(indices)
                label_y = y_sum // len(indices)
                cv2.putText(frame, feature, (label_x, label_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return frame

# Function to Get Landmark Coordinates
def get_landmark_coordinates(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    coordinates = {}

    if results.multi_face_landmarks:
        h, w, _ = frame.shape
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            for feature, (indices, _) in FACIAL_FEATURES.items():
                coordinates[feature] = [(i, int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]

    return coordinates

# Streamlit UI
st.markdown("<h1 style='color: cyan; text-align: left;'>Facial Landmark Detection</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: left; color: #34495E; padding: 10px;'>A Computer Vision Approach for Precise Facial Feature Localization</h2>", unsafe_allow_html=True)

option = st.selectbox("Choose Input Type", ["None", "Camera", "Video", "Image"])

# Camera Processing
if option == "Camera":
    with st.container():
        cap = cv2.VideoCapture(0)
        frame_placeholder = st.empty()
        coords_placeholder = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image")
                break

            processed_frame = process_frame(frame)
            frame_placeholder.image(processed_frame, channels="BGR")

            
            coords = get_landmark_coordinates(frame)
            coords_text = "<br>".join([f"<span style='color: yellow;'>{key}:</span> <span style='color: cyan;'>{coords[key][:3]}</span>" for key in coords])  
            coords_placeholder.markdown(f"""
                    <div style='background-color: #34495E; padding: 10px; border-radius: 10px;'>
                        <h4 style='color: white; text-align: center;'>Facial Landmark Coordinates</h4>
                        {coords_text}
                    </div>
                """, unsafe_allow_html=True)

        cap.release()

# Video Processing
elif option == "Video":
    with st.container():
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            temp_video = tempfile.NamedTemporaryFile(delete=False)
            temp_video.write(uploaded_file.read())

            cap = cv2.VideoCapture(temp_video.name)
            frame_placeholder = st.empty()
            coords_placeholder = st.empty()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                processed_frame = process_frame(frame)
                frame_placeholder.image(processed_frame, channels="BGR")
                coords = get_landmark_coordinates(frame)

                coords_text = "<br>".join([f"<span style='color: yellow;'>{key}:</span> <span style='color: cyan;'>{coords[key][:3]}</span>" for key in coords])  
                coords_placeholder.markdown(f"""
                    <div style='background-color: #34495E; padding: 10px; border-radius: 10px;'>
                        <h4 style='color: white; text-align: center;'>Facial Landmark Coordinates</h4>
                        {coords_text}
                    </div>
                """, unsafe_allow_html=True)


            cap.release()

# Image Processing
elif option == "Image":
    with st.container():
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            processed_image = process_frame(image)
            st.image(processed_image, channels="BGR")
            coords_placeholder = st.empty()

            coords = get_landmark_coordinates(image)
           
            coords_text = "<br>".join([f"<span style='color: yellow;'>{key}:</span> <span style='color: cyan;'>{coords[key][:3]}</span>" for key in coords])  
            coords_placeholder.markdown(f"""
                    <div style='background-color: #34495E; padding: 10px; border-radius: 10px;'>
                        <h4 style='color: white; text-align: center;'>Facial Landmark Coordinates</h4>
                        {coords_text}
                    </div>
                """, unsafe_allow_html=True)
