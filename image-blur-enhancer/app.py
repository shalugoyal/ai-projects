import cv2
from PIL import Image
import numpy as np
from rembg import remove
import os
import shutil
import glob
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
import gradio as gr
import spaces

@spaces.GPU
def cv_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))

@spaces.GPU
def pil_to_cv(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)

@spaces.GPU
def motion_blur(img, distance, amount):
    # Convert to RGBA
    img = img.convert('RGBA')

    # Convert pil to cv
    cv_img = pil_to_cv(img)

    # Generating the kernel
    kernel_motion_blur = np.zeros((distance, distance))
    kernel_motion_blur[int((distance - 1) / 2), :] = np.ones(distance)
    kernel_motion_blur = kernel_motion_blur / distance

    # Applying the kernel to the input image
    output = cv2.filter2D(cv_img, -1, kernel_motion_blur)

    # Convert cv to pil
    blur_img = cv_to_pil(output).convert('RGBA')

    # Blend the original image and the blur image
    final_img = Image.blend(img, blur_img, amount)

    return final_img

@spaces.GPU(enable_queue=True)
def background_motion_blur(background, distance_blur, amount_blur):
    # Remove background
    subject = remove(background)
    amount_subject = 1

    # Blur the background
    background_blur = motion_blur(background, distance_blur, amount_blur)

    # Put the subject on top of the blur background
    subject_on_blur_background = background_blur.copy()
    subject_on_blur_background.paste(background, (0, 0), subject)

    # Blend the subject and the blur background
    result = Image.blend(background_blur, subject_on_blur_background, amount_subject)

    return result

def blur(img, distance_blur, amount_blur):
    return background_motion_blur(img, distance_blur, amount_blur)


title = "Image to Motion Blur"

with gr.Blocks() as motion_api:
    with gr.Column():
        gr.HTML(f"<h1>{title}</h1>")
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(height=500, type='pil', sources=["upload"])
            blur_distance = gr.Slider(label='Blur Distance', minimum=0, maximum=500, value=200)
            blur_amount = gr.Slider(label='Blur Amount', minimum=0.0, maximum=1.0, value=1)
            submit_btn = gr.Button("Run")
        with gr.Column():
            image_output = gr.Image(height=500)

    submit_btn.click(fn=blur, inputs=[image_input, blur_distance, blur_amount], outputs=[image_output])

motion_api.launch(debug=True, show_error=True)
