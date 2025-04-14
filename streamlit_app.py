import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
import tempfile
from decord import VideoReader
import base64
from io import BytesIO
from openai import OpenAI

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
        import openai
        import decord
    except ImportError:
        os.system("pip install openai Pillow matplotlib opencv-python decord")

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

def image_to_base64(image):
    """Convert a PIL image to base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def setup_openrouter_client(api_key):
    """Set up and return the OpenRouter client using provided API key."""
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    return client

def analyze_with_openrouter(client, model_name, video_path, frames_per_second=4, progress_bar=None):
    """Analyze video frames using OpenRouter AI and return response."""
    # Get total duration from video using Decord
    vr = VideoReader(video_path)
    video_fps = vr.get_avg_fps()
    num_frames = len(vr)
    total_duration_sec = round(num_frames / video_fps)
    
    # Create main prompt with instructions
    main_prompt = (
        "You are a fitness expert analyzing a workout video. "
        f"The video is approximately {total_duration_sec} seconds long. "
        f"I'll show you {frames_per_second} frames per second with timestamps. "
        "Your task is to:\n\n"
        
        "1. Identify the exercise being performed\n"
        "2. Count repetitions with timestamps (start/end of each rep)\n"
        "3. Assess tempo (slow/moderate/fast)\n"
        "4. Evaluate form and technique\n\n"
        
        "Please be time-aware in your analysis, tracking movement across frames. "
        "Identify complete repetitions only when the user returns to the starting position. "
        "Note that rep speed may vary during the workout.\n\n"
        
        "I'll now show you the frames in sequence with their timestamps. "
        "Please provide your analysis in this format:\n\n"
        
        "- Exercise identified:\n"
        "- Total repetition count (Detailed with timestamp):\n"
        "- Tempo assessment:\n"
        "- Form evaluation:\n"
        "- Reasoning for your analysis:"
    )
    
    # Extract frames
    with tempfile.TemporaryDirectory() as tmpdir:
        frames = extract_frames(video_path, fps=frames_per_second)
        
        if progress_bar:
            progress_bar.progress(0.2, text="Frames extracted. Processing...")
        
        # Limit frames to prevent API limits
        max_frames = min(len(frames), 16)  # Adjust based on model limits
        frames = frames[:max_frames]
        
        # Prepare message content
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": main_prompt
                    }
                ]
            }
        ]
        
        # Process and add frames to the message
        for i, (timestamp, frame) in enumerate(frames):
            # Process frame (convert to PIL, then to base64)
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_base64 = image_to_base64(img_pil)
            
            # Create a message for each frame with its timestamp
            frame_message = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Frame {i+1} at timestamp {timestamp:.2f} seconds:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_base64}"
                        }
                    }
                ]
            }
            
            messages.append(frame_message)
            
            if progress_bar and i % 4 == 0:
                progress_bar.progress(0.2 + 0.4 * (i / len(frames)), 
                                    text=f"Processing frame {i+1}/{len(frames)}...")
        
        # Add final message requesting analysis
        final_message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Based on all frames shown above, please provide your complete fitness analysis following the format I specified earlier."
                }
            ]
        }
        messages.append(final_message)
    
    if progress_bar:
        progress_bar.progress(0.6, text="Sending to AI for analysis...")
    
    # Try an alternative approach if there are too many separate messages
    if len(messages) > 10:
        # Reformat to a single message with combined content
        combined_content = [{"type": "text", "text": main_prompt}]
        
        for i, (timestamp, frame) in enumerate(frames):
            # Process frame
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_base64 = image_to_base64(img_pil)
            
            # Add frame description
            combined_content.append({"type": "text", "text": f"Frame {i+1} at timestamp {timestamp:.2f} seconds:"})
            
            # Add frame image
            combined_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_base64}"
                }
            })
        
        # Add final request
        combined_content.append({"type": "text", "text": "Based on all frames shown above, please provide your complete fitness analysis following the format I specified earlier."})
        
        # Replace messages with single combined message
        messages = [{
            "role": "user",
            "content": combined_content
        }]

    # Analyze with OpenRouter
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages
        )
        if progress_bar:
            progress_bar.progress(0.9, text="Analysis complete! Preparing results...")
        return completion.choices[0].message.content
    except Exception as e:
        error_msg = f"Error analyzing with OpenRouter: {str(e)}"
        if progress_bar:
            progress_bar.progress(1.0, text="Error occurred during analysis")
        return error_msg

# Sidebar for API key input and model selection
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter your OpenRouter API Key", type="password")
    
    # Model selection
    st.subheader("Select Vision Model")
    model_options = {
        "meta-llama/llama-4-vision": "Llama 4 Vision",
        "anthropic/claude-3-5-sonnet-20240620": "Claude 3.5 Sonnet",
        "anthropic/claude-3-opus-20240229": "Claude 3 Opus",
        "google/gemini-1.5-pro-latest": "Gemini 1.5 Pro",
        "openai/gpt-4o": "GPT-4o",
        "mistralai/mistral-large-latest": "Mistral Large",
        "custom": "Custom Model ID"
    }
    
    selected_model = st.selectbox(
        "Choose a model",
        list(model_options.keys()),
        format_func=lambda x: model_options[x]
    )
    
    # Custom model input
    if selected_model == "custom":
        custom_model = st.text_input("Enter custom model ID")
        if custom_model:
            selected_model = custom_model
    
    frames_per_second = st.slider("Frames per second to extract", min_value=1, max_value=8, value=4)
    st.caption("Higher FPS gives more detailed analysis but takes longer to process")
    
    # Token limit warning
    st.warning("Vision models have token limits. If you get errors, try reducing the frames per second or using a model with higher limits.")
    
    st.markdown("---")
    st.markdown("""
    ### How to use
    1. Enter your OpenRouter API key in the sidebar
    2. Select your preferred model
    3. Upload a fitness video
    4. Click "Analyze Video"
    5. View AI analysis results
    """)
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This app uses OpenRouter.ai to analyze fitness videos.
    It extracts frames from your video and sends them to the selected AI model
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
        model_display_name = model_options.get(selected_model, selected_model)
        st.write(f"Selected model: **{model_display_name}**")
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
            st.error("Please enter your OpenRouter API key in the sidebar.")
        else:
            try:
                # Create progress bar
                progress_text = "Starting analysis..."
                progress_bar = st.progress(0, text=progress_text)
                
                # Setup client with API key
                client = setup_openrouter_client(api_key)
                
                # Extract frames in the background (no display)
                progress_bar.progress(0.1, text="Extracting frames...")
                
                # Analyze with OpenRouter
                analysis_result = analyze_with_openrouter(
                    client, selected_model, temp_file_path, frames_per_second, progress_bar)
                
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
st.caption("AI Fitness Video Analyzer | Powered by OpenRouter.ai")