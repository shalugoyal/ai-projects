# app.py
import os
import torch
import gradio as gr
from TTS.api import TTS

# Ensure Coqui TTS terms are accepted
os.environ["COQUI_TOS_AGREED"] = "1"

# Determine runtime device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load XTTS model from Coqui (multilingual + speaker cloning)
voice_cloner = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(DEVICE)

def generate_cloned_audio(text_input: str, reference_audio_path: str) -> str:
    """
    Generate an audio file by cloning the voice in reference_audio_path and synthesizing text_input.
    
    Args:
        text_input (str): The text to synthesize.
        reference_audio_path (str): File path to reference speaker audio.

    Returns:
        str: Path to the generated audio file.
    """
    output_audio_path = "cloned_voice.wav"
    voice_cloner.tts_to_file(
        text=text_input,
        speaker_wav=reference_audio_path,
        language="en",
        file_path=output_audio_path
    )
    return output_audio_path

# Gradio UI setup
def launch_voice_cloner_ui():
    demo = gr.Interface(
        fn=generate_cloned_audio,
        inputs=[
            gr.Textbox(label="Input Text", placeholder="Enter the message..."),
            gr.Audio(type="filepath", label="Voice Reference (.wav or .mp3)")
        ],
        outputs=gr.Audio(type="filepath", label="Cloned Voice Output"),
        title="🎙️ EchoMorph AI - Neural Voice Cloner",
        description="""
        Generate realistic cloned speech from a short voice sample.
        Upload a reference voice file and enter the message to be spoken.
        Powered by Coqui XTTS v2.
        """,
        theme=gr.themes.Base(primary_hue="cyan", neutral_hue="slate"),
        examples=[
            ["Hello! This is my voice clone speaking.", "audio/Jeff-Goldblum.mp3"],
            ["I can say anything in your voice!", "audio/Megan-Fox.mp3"],
        ]
    )
    demo.launch()

if __name__ == "__main__":
    launch_voice_cloner_ui()
