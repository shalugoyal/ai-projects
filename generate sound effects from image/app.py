# visionwave.py

import os
import tempfile
import gradio as gr
from dotenv import load_dotenv
import torch
from scipy.io.wavfile import write
from diffusers import DiffusionPipeline
from transformers import pipeline

# Load Hugging Face token from .env file
load_dotenv()
hf_token = os.getenv("HF_TKN")  # required for accessing AudioLDM2

# Set GPU if available
device_id = 0 if torch.cuda.is_available() else -1

# Image captioning model (Vision Transformer + GPT-2)
captioning_pipeline = pipeline(
    "image-to-text",
    model="nlpconnect/vit-gpt2-image-captioning",
    device=device_id
)

# Load AudioLDM2 model (for converting text to sound)
audio_pipe = DiffusionPipeline.from_pretrained(
    "cvssp/audioldm2",
    use_auth_token=hf_token
)

def describe_image(image_file):
    """
    Generates a caption for the uploaded image.
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(image_file)
            image_path = tmp.name

        results = captioning_pipeline(image_path)
        caption = results[0].get("generated_text", "").strip()
        return caption if caption else "No caption generated."
    except Exception as e:
        return f"Error: {e}"

def generate_audio_from_caption(caption):
    """
    Uses AudioLDM2 to generate a sound effect based on the image caption.
    """
    try:
        audio_pipe.to("cuda")
        result = audio_pipe(prompt=caption, num_inference_steps=50, guidance_scale=7.5)
        audio_pipe.to("cpu")
        audio = result.audios[0]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            write(tmp_wav.name, 16000, audio)
            return tmp_wav.name
    except Exception as e:
        return f"Audio generation error: {e}"

# -------- UI SECTION --------

css = "#col-container{margin: 0 auto; max-width: 800px;}"

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML("<h1 style='text-align: center;'>🎨 VisionWave: Image-to-Sound Generator</h1>")
        gr.Markdown("""
            Upload an image and listen to its story. This app:
            - Describes the image
            - Synthesizes sound effects to match the scene
            Powered by Hugging Face models.
        """)

    # Inputs and Outputs
    image_input = gr.File(label="📸 Upload Image", type="binary")
    generate_caption_btn = gr.Button("📝 Generate Description")
    caption_output = gr.Textbox(label="Generated Caption", interactive=False)

    generate_audio_btn = gr.Button("🎧 Generate Sound Effect")
    audio_output = gr.Audio(label="Generated Audio")

    # Event Handlers
    generate_caption_btn.click(fn=describe_image, inputs=image_input, outputs=caption_output)
    generate_audio_btn.click(fn=generate_audio_from_caption, inputs=caption_output, outputs=audio_output)

demo.launch(debug=True, share=True)
