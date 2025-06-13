import gradio as gr
from gradio_imageslider import ImageSlider  # For showing side-by-side comparisons
from loadimg import load_img  # Local utility to handle image loading from file or URL
import spaces  # Required to use Hugging Face Spaces features
from transformers import AutoModelForImageSegmentation  # To load segmentation model from HF
import torch
from torchvision import transforms

# Set matrix multiplication precision for better performance (recommended for GPUs)
torch.set_float32_matmul_precision("high")

# Load BiRefNet model for background segmentation from Hugging Face
birefnet = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet", trust_remote_code=True
)
birefnet.to("cuda")  # Send model to GPU

# Define preprocessing steps for input image
transform_image = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

@spaces.GPU  # Use GPU on Hugging Face Spaces
# Background removal function
# Takes an image input, returns a tuple (segmented image, original image)
def remove_background(image):
    im = load_img(image, output_type="pil")  # Load and convert to PIL
    im = im.convert("RGB")
    image_size = im.size  # Store original size for resizing mask later
    origin = im.copy()  # Keep a copy of the original image

    image = load_img(im)
    input_images = transform_image(image).unsqueeze(0).to("cuda")  # Apply transforms and send to GPU

    # Run inference
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()  # Get prediction and move to CPU

    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)  # Convert tensor to image
    mask = pred_pil.resize(image_size)  # Resize mask to original image size
    image.putalpha(mask)  # Add alpha channel as background mask
    return (image, origin)  # Return masked and original image

# Define Gradio image sliders to show before/after comparison
slider1 = ImageSlider(label="BiRefNet Result", type="pil")
slider2 = ImageSlider(label="BiRefNet Result", type="pil")

# Input options
image = gr.Image(label="Upload an Image")
text = gr.Textbox(label="Paste an Image URL")

# Example input images
chameleon = load_img("chameleon.jpg", output_type="pil")
url = "https://hips.hearstapps.com/hmg-prod/images/gettyimages-1229892983-square.jpg"

# Create Gradio interface for file upload
tab1 = gr.Interface(
    fn=remove_background,
    inputs=image,
    outputs=slider1,
    examples=[chameleon],
    api_name="image"  # Optional: for Gradio API support
)

# Create Gradio interface for image URL input
tab2 = gr.Interface(
    fn=remove_background,
    inputs=text,
    outputs=slider2,
    examples=[url],
    api_name="text"
)

# Combine interfaces into tabbed layout
app = gr.TabbedInterface(
    [tab1, tab2],
    ["Upload Image", "Use URL"],
    title="BiRefNet Background Removal"
)

# Launch Gradio app
if __name__ == "__main__":
    app.launch()