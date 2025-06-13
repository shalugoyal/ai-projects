import gradio as gr
import time
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load BLIP image captioning model and processor
# Model used: Salesforce/blip-image-captioning-large (from Hugging Face)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

def caption_image(img_path, min_len, max_len):
    """
    Generates a caption for the provided image.

    Args:
        img_path: File path of the uploaded image.
        min_len: Minimum length of the generated caption.
        max_len: Maximum length of the generated caption.

    Returns:
        A string containing the generated caption.
    """
    image = Image.open(img_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")

    outputs = model.generate(**inputs, min_length=min_len, max_length=max_len)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

def timed_caption(img_path, min_len, max_len):
    """
    Captions image and records execution time.
    """
    start = time.time()
    caption = caption_image(img_path, min_len, max_len)
    end = time.time()
    duration = round(end - start, 2)
    return f"{caption}\n\n⏱️ Inference Time: {duration} seconds"

# Gradio UI
iface = gr.Interface(
    fn=timed_caption,
    title="🖼️ BLIP Image Captioning",
    description="Generate natural language captions for images using the BLIP model from Hugging Face.",
    inputs=[
        gr.Image(type="filepath", label="Upload Image"),
        gr.Slider(minimum=1, maximum=100, value=30, label="Minimum Caption Length"),
        gr.Slider(minimum=1, maximum=100, value=100, label="Maximum Caption Length")
    ],
    outputs=gr.Textbox(label="Generated Caption")
)

if __name__ == "__main__":
    iface.launch()
