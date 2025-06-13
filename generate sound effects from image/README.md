# 🎨 VisionWave: AI-Powered Image-to-Sound Generator

Transform your images into realistic sound effects using state-of-the-art AI models. VisionWave takes a single image, generates a caption, and creates matching audio — all in seconds.

## 🚀 Features

- 🧠 AI-generated image descriptions (ViT-GPT2)
- 🔊 Sound effect generation (AudioLDM2)
- 🖼️ Simple drag-and-drop interface (Gradio)
- 🔒 Token-authenticated access to models via Hugging Face

---

## 🛠️ How It Works

1. **Upload an image** — (PNG, JPG)
2. **Generate Description** — Uses `vit-gpt2` to caption your image
3. **Generate Sound** — Uses `audioldm2` to create a matching ambient sound

---

<!-- To use this application, you’ll need a Hugging Face access token with access to gated models.

🎯 Required Model
cvssp/audioldm2 (requires authentication)

✅ How to Get Your Token
Visit: https://huggingface.co/settings/tokens

Click "New token"

Name it (e.g., visionwave)

Set the Role to read

Copy the token -->