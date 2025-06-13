# BiRefNet Background Remover 🎯

This project provides a clean web-based interface to perform **background removal** using the [BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet) image segmentation model.

## 🚀 Features

- Remove backgrounds using state-of-the-art BiRefNet
- Supports file upload and URL input
- Side-by-side image comparison
- Deployed-ready on Hugging Face Spaces (Gradio)

## 🔧 How It Works

1. Load your image or paste a URL
2. The model generates a segmentation mask
3. The mask is applied to create a transparent background

## 🧠 Model Used

- Model: [`ZhengPeng7/BiRefNet`](https://huggingface.co/ZhengPeng7/BiRefNet)
- Task: Image Segmentation
- Hosted on: Hugging Face (no token required)

## 💻 Tech Stack

- Python 🐍
- [Transformers](https://github.com/huggingface/transformers)
- [Gradio](https://gradio.app)
- [Torchvision](https://pytorch.org/vision/)
