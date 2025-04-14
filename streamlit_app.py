import streamlit as st
import os
import cv2
import numpy as np
import google.generativeai as genai
import matplotlib.pyplot as plt
from PIL import Image
import tempfile
import math
import time
from decord import VideoReader
import base64
from io import BytesIO

# Set page config
st.set_page_config(
    page_title="Fitness Video Analyzer",
    page_icon="💪",
    layout="wide",
)

# App title and description
st.title("🏋️ AI Fitness Video Analyzer")
st.markdown("""
This app analyzes fitness videos to identify exercises, count repetitions, assess tempo, and evaluate form.
Simply upload a video of your workout, and our AI will provide detailed feedback.
""")

# Function to install required packages (for notebook environments)
def install_required_packages():
    try:
        import google.generativeai
        import decord
    except ImportError:
        os.system("pip install google-generativeai Pillow matplotlib opencv-python decord")

# Attempt to install packages if in a supported environment
try:
    install_required_packages()
except:
    st.warning("Unable to automatically install required packages. If you encounter errors, please install manually.")

# Resize frame function
def smart_resize(frame, min_dim=300, max_dim=400):
    """Resize frame so the longest side is between min_dim and max_dim."""
    h, w = frame.shape[:2]
    longer = max(h, w)
    scale = min(max_dim, max(min_dim, longer)) / longer
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(frame, new_size)

def draw_timestamp(frame, timestamp):
    """Overlay timestamp text on the frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"Time: {timestamp:.2f}s"
    cv2.putText(frame, text, (5, 20), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return frame

def extract_frames(video_path, fps=4, min_dim=300, max_dim=400):
    """
    Extracts N frames per second, resizes them smartly, and adds timestamps.
    Returns individual frames instead of a grid.
    """
    vr = VideoReader(video_path)
    video_fps = vr.get_avg_fps()
    num_frames = len(vr)
    duration = num_frames / video_fps
    total_seconds = int(duration)
    
    frames_with_timestamps = []
    
    for sec in range(total_seconds):
        for i in range(fps):
            timestamp = sec + i / fps
            frame_index = int(timestamp * video_fps)
            if frame_index < num_frames:
                frame = vr[frame_index].asnumpy()
                frame = smart_resize(frame, min_dim, max_dim)
                frame = draw_timestamp(frame, timestamp)
                frames_with_timestamps.append((timestamp, frame))
    
    return frames_with_timestamps  # List of tuples: (timestamp, frame)

def setup_gemini_api(api_key, model_name):
    """Set up and return the Gemini model using provided API key and model name."""
    genai.configure(api_key=api_key)

    # Set up the model
    generation_config = {
        "temperature": 0.1,
        "top_p": 0.1,
        "top_k": 1,
    }

    # Use the user-selected model
    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config=generation_config
    )

    return model

def analyze_with_gemini_individual_frames(model, video_path, frames_per_second=4, progress_bar=None):
    """Analyze video frames using Gemini AI and return response."""
    # Role definition
    Role = (
        "You're a fitness expert and your task is to analyze fitness videos. "
        "You need to be aware of the user's movement during the sequence of frames extracted from the video. "
        "Each frame is annotated with the timestamp showing its exact position in the video."
        "Write shortly"
    )

    # Get total duration from video using Decord
    vr = VideoReader(video_path)
    video_fps = vr.get_avg_fps()
    num_frames = len(vr)
    total_duration_sec = round(num_frames / video_fps)

    # Task prompt with total video time
    TaskPrompt = (
        f"You will be given a sequence of individual frames extracted from a fitness video. "
        f"The total duration of the video is approximately {total_duration_sec} seconds. "
        f"I'm showing you {frames_per_second} frames per second, and every frame is annotated with a timestamp showing its exact position in the video.\n\n"

        "Your task is to analyze the user's movement by reviewing the frames in order, with close attention to the timestamps. Your analysis must be time-aware, not just based on visual similarities.\n\n"

        "The analysis consists of the following steps:\n\n"

        "1. **Exercise Identification**:\n"
        "- First, identify which exercise is being performed (e.g., squats, push-ups, lunges).\n"
        "- Use full-body positioning, involved joints, angles, and movement patterns over time to determine the exercise type.\n"
        "- Do not rely on the visual appearance of a single frame. Your identification must be based on how the user's body moves across time.\n\n"

        "2. **Repetition Counting**:\n"
        "- After identifying the exercise, determine when each repetition begins and ends.\n"
        "- A repetition is only considered complete when the user returns to the starting position of the movement.\n"
        "- It's especially important to correctly identify where in a rep the video starts — the beginning, middle, or end.\n"
        "- To do this, go beyond visual comparison and conduct a **tempospatial analysis**: recognize which joints and body parts are involved in the exercise, then track their positioning and movement over time using timestamps.\n"
        "- Compare body posture across frames and infer movement trajectories to accurately detect rep boundaries.\n"
        "- Always look ahead to the next frames to confirm if a rep has truly ended. These are static frames sampled from a continuous video — temporal context is essential.\n"
        "- Keep in mind that rep speed may vary during the workout. For example, the user might perform the first 3 reps at 2 seconds per rep, then slow down to 3 seconds or speed up to 1 second per rep.\n"
        "- **Never assume a fixed tempo or repetition pattern based on earlier reps**. Each rep must be analyzed independently based on timestamped frames.\n\n"

        "3. **Tempo Analysis**:\n"
        "- For each repetition, calculate its duration using the frame timestamps.\n"
        "- Categorize the tempo as slow, moderate, or fast based on time.\n"
        "- If the user changes tempo across the session, explicitly highlight this change and explain when and how it happens.\n\n"

        "4. **Form & Technique Evaluation**:\n"
        "- Throughout the exercise, assess the user's form: body posture, joint alignment, range of motion, balance, and stability.\n"
        "- Determine whether the user is performing the movement correctly, based on the standards of the identified exercise.\n"
        "- Flag any issues such as incomplete movement, poor control, or risky alignment (e.g., knees going too far forward, rounded back, lack of full extension).\n\n"

        "**Key Instructions:**\n"
        "- Always refer to specific frames and timestamps when making observations.\n"
        "- Your reasoning must be time-aware — describe how the user's position evolves across time.\n"
        "- Avoid assumptions. If parts of the movement are unclear or missing from the video sample, explicitly state the limitations.\n"
        "- Use clear biomechanical and fitness-specific terminology wherever appropriate.\n\n"

        "Your response should reflect a detailed, timestamp-driven analysis of movement and should clearly tie all conclusions back to specific moments in the video."
    )

    # Output format
    OutputFormat = (
        "Please respond in this format:\n"
        "- Exercise identified:\n"
        "- Total repetition count (Detailed with timestamp):\n"
        "- Tempo assessment:\n"
        "- Form evaluation:\n"
        "- Reasoning for your analysis:"
    )

    Content = []
    
    # Add role and task prompt
    Content.append(Role)
    Content.append(TaskPrompt)

    # Extract frames
    with tempfile.TemporaryDirectory() as tmpdir:
        frame_paths = []
        frames = extract_frames(video_path, fps=frames_per_second)
        
        if progress_bar:
            progress_bar.progress(0.2, text="Frames extracted. Processing...")

        # Maximum number of frames to process (adjust based on model limits)
        max_frames = min(len(frames), 32)  # Limit to prevent exceeding API context
        frames = frames[:max_frames]
        
        for i, (timestamp, frame) in enumerate(frames):
            frame_path = os.path.join(tmpdir, f"frame_{i:04d}_{timestamp:.2f}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append((timestamp, frame_path))

        # Add frames to content with timestamps
        for i, (timestamp, path) in enumerate(frame_paths):
            img = Image.open(path)
            Content.append(img)
            Content.append(
                f"Frame {i+1}: This frame is at timestamp {timestamp:.2f} seconds of the video. "
                f"Please analyze this frame in the context of the sequence."
            )
            
            if progress_bar and i % 4 == 0:
                progress_bar.progress(0.2 + 0.4 * (i / len(frame_paths)), 
                                    text=f"Processing frame {i+1}/{len(frame_paths)}...")

    # Add output instructions
    Content.append(OutputFormat)
    
    if progress_bar:
        progress_bar.progress(0.6, text="Sending to AI for analysis...")

    # Analyze with Gemini
    try:
        response = model.generate_content(Content)
        if progress_bar:
            progress_bar.progress(0.9, text="Analysis complete! Preparing results...")
        return response.text, Content
    except Exception as e:
        error_msg = f"Error analyzing with Gemini: {str(e)}"
        if progress_bar:
            progress_bar.progress(1.0, text="Error occurred during analysis")
        return error_msg, Content

# Sidebar for API key input and model selection
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter your Gemini API Key", type="password")
    
    # Model selection
    st.subheader("Select Gemini Model")
    model_options = {
        "gemini-pro-vision": "Gemini Pro Vision",
        "gemini-1.5-pro-vision": "Gemini 1.5 Pro Vision",
        "gemini-1.5-flash-vision": "Gemini 1.5 Flash Vision",
        "gemini-2.0-pro-vision": "Gemini 2.0 Pro Vision (if available)",
        "gemini-2.5-pro-exp-03-25": "Gemini 2.5 Pro Experimental"
    }
    selected_model = st.selectbox(
        "Choose a model",
        list(model_options.keys()),
        format_func=lambda x: model_options[x]
    )
    
    frames_per_second = st.slider("Frames per second to extract", min_value=1, max_value=8, value=4)
    st.caption("Higher FPS gives more detailed analysis but takes longer to process")
    
    st.markdown("---")
    st.markdown("""
    ### How to use
    1. Enter your Gemini API key in the sidebar
    2. Select your preferred Gemini model
    3. Upload a fitness video
    4. Click "Analyze Video"
    5. View AI analysis results
    """)
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This app uses Google's Gemini AI to analyze fitness videos.
    It extracts frames from your video and sends them to the AI model
    for detailed exercise analysis.
    """)

# Main interface
uploaded_file = st.file_uploader("Upload a fitness video", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    # Save uploaded file to a temporary location
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_file.write(uploaded_file.read())
    temp_file_path = temp_file.name
    temp_file.close()
    
    # Display the video
    st.video(temp_file_path)
    
    # Analysis section with collapsible details
    with st.expander("Video Analysis Details", expanded=False):
        st.write(f"Selected model: **{model_options[selected_model]}**")
        st.write(f"Frame extraction rate: **{frames_per_second} frames per second**")
        vr = VideoReader(temp_file_path)
        video_fps = vr.get_avg_fps()
        num_frames = len(vr)
        duration = num_frames / video_fps
        st.write(f"Video duration: **{duration:.2f} seconds**")
        st.write(f"Original video FPS: **{video_fps:.2f}**")
    
    # Analyze button
    if st.button("Analyze Video"):
        if not api_key:
            st.error("Please enter your Gemini API key in the sidebar.")
        else:
            try:
                # Create progress bar
                progress_text = "Starting analysis..."
                progress_bar = st.progress(0, text=progress_text)
                
                # Setup model with API key and selected model
                model = setup_gemini_api(api_key, selected_model)
                
                # Extract frames in the background (no display)
                progress_bar.progress(0.1, text="Extracting frames...")
                
                # Analyze with Gemini
                analysis_result, _ = analyze_with_gemini_individual_frames(
                    model, temp_file_path, frames_per_second, progress_bar)
                
                # Display results
                st.subheader("AI Analysis Results")
                st.markdown(analysis_result)
                
                # Complete progress
                progress_bar.progress(1.0, text="Analysis complete!")
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
            finally:
                # Clean up the temporary file
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
else:
    # Display demo content when no file is uploaded
    st.info("Please upload a fitness video to begin analysis.")
    
    # Sample analysis display
    with st.expander("See example analysis"):
        st.markdown("""
        ### Example Analysis
        
        - **Exercise identified**: Bodyweight Squat
        
        - **Total repetition count (Detailed with timestamp)**:
          - Rep 1: 0.00s - 2.50s
          - Rep 2: 2.50s - 5.25s
          - Rep 3: 5.25s - 8.00s
          - Rep 4: 8.00s - 10.75s
          - Rep 5: 10.75s - 13.50s (partial rep)
        
        - **Tempo assessment**:
          - Average tempo: Moderate (approximately 2.5 seconds per rep)
          - Consistent timing throughout all completed repetitions
        
        - **Form evaluation**:
          - Good depth on all squats
          - Knees track over toes appropriately
          - Core remains engaged throughout
          - Minor forward lean at bottom position
          - Suggestion: Focus on maintaining more upright torso
        
        - **Reasoning for your analysis**:
          Analysis based on tracking hip and knee angles across timestamps. The exercise was identified as a bodyweight squat based on the characteristic downward/upward movement pattern with weight on both legs. Repetitions were counted by identifying the starting position (standing tall) and tracking the full movement cycle back to that position. Form assessment considered knee tracking, torso angle, and depth of the squat.
        """)

# Footer
st.markdown("---")
st.caption("AI Fitness Video Analyzer | Powered by Gemini AI")