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
import toml
from decord import VideoReader
import base64
from io import BytesIO
import gdown
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.oauth2.service_account import Credentials
from google.auth.transport.requests import Request

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
        import pydrive
        import gdown
    except ImportError:
        os.system("pip install google-generativeai Pillow matplotlib opencv-python decord pydrive2 gdown")

# Attempt to install packages if in a supported environment
try:
    install_required_packages()
except:
    st.warning("Unable to automatically install required packages. If you encounter errors, please install manually.")

# Load default configuration from toml file
@st.cache_resource
def load_config():
    try:
        if os.path.exists("config.toml"):
            return toml.load("config.toml")
        else:
            return {"gemini_api_key": "", "google_drive_credentials": {}}
    except Exception as e:
        st.error(f"Error loading configuration: {e}")
        return {"gemini_api_key": "", "google_drive_credentials": {}}

config = load_config()

# Google Drive Authentication
def authenticate_google_drive(service_account_info=None):
    """Authenticate with Google Drive using service account or OAuth."""
    try:
        if service_account_info:
            # Use service account
            credentials = Credentials.from_service_account_info(
                service_account_info,
                scopes=['https://www.googleapis.com/auth/drive']
            )
            drive = GoogleDrive(credentials)
            return drive
        else:
            # Use OAuth flow
            gauth = GoogleAuth()
            gauth.LocalWebserverAuth()  # Creates local webserver and auto handles authentication
            drive = GoogleDrive(gauth)
            return drive
    except Exception as e:
        st.error(f"Error authenticating with Google Drive: {e}")
        return None

# Upload file to Google Drive
def upload_to_drive(drive, file_path, folder_id=None):
    """Upload file to Google Drive and return file ID."""
    try:
        file_drive = drive.CreateFile({'title': os.path.basename(file_path)})
        if folder_id:
            file_drive['parents'] = [{'id': folder_id}]
        file_drive.SetContentFile(file_path)
        file_drive.Upload()
        return file_drive['id']
    except Exception as e:
        st.error(f"Error uploading to Google Drive: {e}")
        return None

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

def extract_frames(video_path, fps=4, min_dim=300, max_dim=400, use_gdrive=False, drive=None, folder_id=None):
    """
    Extracts N frames per second, resizes them smartly, and adds timestamps.
    Returns individual frames instead of a grid.
    Can optionally save to Google Drive.
    """
    vr = VideoReader(video_path)
    video_fps = vr.get_avg_fps()
    num_frames = len(vr)
    duration = num_frames / video_fps
    total_seconds = int(duration)
    
    frames_with_timestamps = []
    
    # Create temporary directory for frames
    with tempfile.TemporaryDirectory() as tmpdir:
        for sec in range(total_seconds):
            for i in range(fps):
                timestamp = sec + i / fps
                frame_index = int(timestamp * video_fps)
                if frame_index < num_frames:
                    frame = vr[frame_index].asnumpy()
                    frame = smart_resize(frame, min_dim, max_dim)
                    frame = draw_timestamp(frame, timestamp)
                    
                    # Save frame locally
                    frame_filename = f"frame_{sec}_{i}_{timestamp:.2f}.jpg"
                    frame_path = os.path.join(tmpdir, frame_filename)
                    cv2.imwrite(frame_path, frame)
                    
                    # If using Google Drive, upload frame
                    if use_gdrive and drive:
                        file_id = upload_to_drive(drive, frame_path, folder_id)
                        frames_with_timestamps.append((timestamp, frame_path, file_id))
                    else:
                        frames_with_timestamps.append((timestamp, frame_path))
    
    return frames_with_timestamps

def setup_gemini_api(api_key, model_name, temperature=0.1, top_p=0.1, top_k=1):
    """Set up and return the Gemini model using provided API key and model name."""
    genai.configure(api_key=api_key)

    # Set up the model with user-configurable parameters
    generation_config = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
    }

    # Use the user-selected model
    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config=generation_config
    )

    return model

def analyze_with_gemini_individual_frames(model, video_path, frames_per_second=4, progress_bar=None, 
                                         use_gdrive=False, drive=None, folder_id=None):
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

    # Extract frames and optionally save to Google Drive
    with tempfile.TemporaryDirectory() as tmpdir:
        if progress_bar:
            progress_bar.progress(0.1, text="Extracting frames...")
            
        # Extract frames with optional Google Drive storage
        frames = extract_frames(
            video_path, 
            fps=frames_per_second, 
            use_gdrive=use_gdrive, 
            drive=drive, 
            folder_id=folder_id
        )
        
        if progress_bar:
            progress_bar.progress(0.2, text="Frames extracted. Processing...")

        # Maximum number of frames to process (adjust based on model limits)
        max_frames = min(len(frames), 32)  # Limit to prevent exceeding API context
        frames = frames[:max_frames]
        
        # Process frames - different handling for GDrive vs local
        for i, frame_data in enumerate(frames):
            if use_gdrive and drive:
                timestamp, frame_path, file_id = frame_data
                img = Image.open(frame_path)
            else:
                timestamp, frame_path = frame_data
                img = Image.open(frame_path)
                
            Content.append(img)
            Content.append(
                f"Frame {i+1}: This frame is at timestamp {timestamp:.2f} seconds of the video. "
                f"Please analyze this frame in the context of the sequence."
            )
            
            if progress_bar and i % 4 == 0:
                progress_bar.progress(0.2 + 0.4 * (i / len(frames)), 
                                    text=f"Processing frame {i+1}/{len(frames)}...")

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

# Create a folder in Google Drive
def create_drive_folder(drive, folder_name="Fitness_Video_Analysis"):
    """Create a folder in Google Drive and return its ID."""
    try:
        folder = drive.CreateFile({
            'title': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        })
        folder.Upload()
        return folder['id']
    except Exception as e:
        st.error(f"Error creating Google Drive folder: {e}")
        return None

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # API Key Configuration
    st.subheader("API Key Settings")
    use_default_key = st.checkbox("Use default API key from config", 
                                value=True if config.get("gemini_api_key") else False)
    
    if use_default_key and config.get("gemini_api_key"):
        api_key = config.get("gemini_api_key")
        st.success("Using default API key from config.toml")
    else:
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
    
    # Advanced settings
    st.subheader("Advanced Settings")
    with st.expander("Model Parameters"):
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.1,
                             help="Higher values make output more random, lower values more deterministic")
        top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.1, step=0.1,
                       help="Controls diversity via nucleus sampling")
        top_k = st.slider("Top K", min_value=1, max_value=40, value=1,
                       help="Controls diversity by limiting to top k tokens")
    
    frames_per_second = st.slider("Frames per second to extract", min_value=1, max_value=8, value=4)
    st.caption("Higher FPS gives more detailed analysis but takes longer to process")
    
    # Storage options
    st.subheader("Storage Options")
    use_gdrive = st.checkbox("Use Google Drive for frame storage", value=False)
    if use_gdrive:
        st.info("You'll need to authenticate with Google Drive when processing")
        gdrive_folder_name = st.text_input("Google Drive folder name", value="Fitness_Video_Analysis")
    
    st.markdown("---")
    st.markdown("""
    ### How to use
    1. Choose API key option
    2. Select your preferred Gemini model
    3. Adjust parameters if needed
    4. Upload a fitness video
    5. Click "Analyze Video"
    6. View AI analysis results
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
        st.write(f"Temperature: **{temperature}**")
        st.write(f"Top P: **{top_p}**")
        st.write(f"Top K: **{top_k}**")
        st.write(f"Using Google Drive: **{'Yes' if use_gdrive else 'No'}**")
        
        vr = VideoReader(temp_file_path)
        video_fps = vr.get_avg_fps()
        num_frames = len(vr)
        duration = num_frames / video_fps
        st.write(f"Video duration: **{duration:.2f} seconds**")
        st.write(f"Original video FPS: **{video_fps:.2f}**")
    
    # Analyze button
    if st.button("Analyze Video"):
        if not api_key:
            st.error("Please enter your Gemini API key in the sidebar or ensure a default key is in config.toml.")
        else:
            try:
                # Create progress bar
                progress_text = "Starting analysis..."
                progress_bar = st.progress(0, text=progress_text)
                
                # Initialize Google Drive if selected
                drive = None
                folder_id = None
                if use_gdrive:
                    progress_bar.progress(0.05, text="Authenticating with Google Drive...")
                    drive = authenticate_google_drive(config.get("google_drive_credentials"))
                    if drive:
                        folder_id = create_drive_folder(drive, gdrive_folder_name)
                        if folder_id:
                            st.success(f"Created folder in Google Drive: {gdrive_folder_name}")
                        else:
                            st.warning("Couldn't create Google Drive folder. Using local storage instead.")
                            use_gdrive = False
                    else:
                        st.warning("Google Drive authentication failed. Using local storage instead.")
                        use_gdrive = False
                
                # Setup model with API key and selected model
                model = setup_gemini_api(api_key, selected_model, temperature, top_p, top_k)
                
                # Analyze with Gemini
                analysis_result, _ = analyze_with_gemini_individual_frames(
                    model, temp_file_path, frames_per_second, progress_bar,
                    use_gdrive=use_gdrive, drive=drive, folder_id=folder_id
                )
                
                # Display results
                st.subheader("AI Analysis Results")
                st.markdown(analysis_result)
                
                # Complete progress
                progress_bar.progress(1.0, text="Analysis complete!")
                
                # Provide link to Google Drive folder if used
                if use_gdrive and folder_id:
                    st.info(f"Frame images stored in Google Drive folder: {gdrive_folder_name}")
                
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