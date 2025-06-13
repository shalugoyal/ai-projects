# Import required libraries
import streamlit as st
from rembg import remove              # Used for background removal from images
from PIL import Image                 # Python Imaging Library for image handling
from io import BytesIO                # For in-memory file operations
import numpy as np                   # For array manipulations
import base64                        # For encoding image for download
import random                        # For shuffling image search results
import cv2                           # OpenCV for image processing (pencil sketch)
import requests                      # For HTTP requests (used in image search)
from bs4 import BeautifulSoup        # For parsing HTML (used in image search)

# -------------------- App Configuration --------------------
st.set_page_config(page_title="ImageMagic - Your Image Processing Companion")
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB limit for uploaded files

# -------------------- Image Processing Functions --------------------

def remove_background(image):
    """
    Removes the background from an image using the 'rembg' library.

    Parameters:
        image (PIL.Image): Input image.

    Returns:
        PIL.Image: Image with background removed.
    """
    return remove(image)

def convert_to_pencil_sketch(image):
    """
    Converts a BGR image to a pencil sketch using OpenCV.

    Parameters:
        image (numpy.ndarray): OpenCV image (BGR format).

    Returns:
        PIL.Image: Pencil sketch image.
    """
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    invert = cv2.bitwise_not(gray_img)
    blur = cv2.GaussianBlur(invert, (111, 111), 0)
    invertedblur = cv2.bitwise_not(blur)
    sketch = cv2.divide(gray_img, invertedblur, scale=256.0)
    return Image.fromarray(sketch)

def convert_image(img):
    """
    Converts a PIL image to byte stream for download.

    Parameters:
        img (PIL.Image): Image to convert.

    Returns:
        bytes: Byte stream of image.
    """
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def get_image_download_link(img, filename="result.png"):
    """
    Generates a base64-encoded download link for an image.

    Parameters:
        img (PIL.Image): Image to encode and download.
        filename (str): Download filename.

    Returns:
        str: HTML anchor tag with base64-encoded download link.
    """
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{img_str}" download="{filename}">Download Image</a>'
    return href

def process_image(upload, action):
    """
    Handles uploaded image and applies selected action.

    Parameters:
        upload: Uploaded image file.
        action: Action selected by user from sidebar.
    """
    image = Image.open(upload)

    coli = st.columns(5)
    with coli[0]:
        st.image(image, caption="Original Image", width=300)
    with coli[3]:
        if action == "Remove Background":
            result = remove_background(image)
            st.image(result, caption="Background Removed", width=300)
        elif action == "Convert to Pencil Sketch":
            result = convert_to_pencil_sketch(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
            st.image(result, caption="Pencil Sketch", width=300)

    st.markdown(get_image_download_link(result), unsafe_allow_html=True)

def search_unsplash_images(query):
    """
    Searches for images on Unsplash by scraping the website.

    ⚠️ Not using the official Unsplash API. This is for demonstration only.

    Parameters:
        query (str): Search keyword.
    """
    try:
        url = f"https://unsplash.com/s/photos/{query}"
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Scrape image URLs
        image_urls = [img['src'] for img in soup.find_all('img') if img.get('src')]
        image_urls = image_urls[:40]
        random.shuffle(image_urls)

        # Display images in columns
        num_columns = 3
        images_per_column = len(image_urls) // num_columns
        for i in range(num_columns):
            col = st.columns(num_columns)
            start_index = i * images_per_column
            end_index = (i + 1) * images_per_column
            for j, image_url in enumerate(image_urls[start_index:end_index], start=start_index):
                image_data = requests.get(image_url).content
                image = Image.open(BytesIO(image_data))
                col[j % num_columns].image(image, use_column_width=True)

    except requests.exceptions.RequestException:
        st.warning("Images could not be loaded. Please try again.")

# -------------------- Main App UI --------------------

# App Title and Description
st.title("🎉 Welcome to ImageMagic")
st.write("""
✨ ImageMagic is your all-in-one tool for basic image editing tasks.
Upload an image and choose what you'd like to do: Remove the background or convert it to a pencil sketch!
""")

# Sidebar Options
st.sidebar.title("Options")
selected_option = st.sidebar.radio("Select an option", ("Home", "Remove Background", "Convert to Pencil Sketch", "Search image", "README.md"))

# -------------------- Page Logic --------------------

if selected_option == "README.md":
    # In-app documentation
    st.title("📘 README")
    readme_content = """
    ## ImageMagic

    A simple yet powerful image editing app built using Python and Streamlit.

    ### 🔧 Features:
    - Background removal using the `rembg` library.
    - Pencil sketch generation using OpenCV.
    - Image search feature (via Unsplash HTML scraping).
    - Download processed images directly.

    ### 🛠️ How to Run Locally:
    ```
    git clone <your-github-repo-url>
    cd ImageMagic
    pip install -r requirements.txt
    streamlit run app.py
    ```

    ### 📦 Dependencies:
    - Streamlit
    - OpenCV
    - rembg
    - requests
    - BeautifulSoup
    - Pillow
    - numpy

    ### 📌 Note:
    The image search is for demonstration purposes only. For production, please use the [Unsplash API](https://unsplash.com/developers).

    ### 📄 License:
    MIT License
    """
    st.markdown(readme_content)

elif selected_option == "Home":
    st.subheader("🏠 Home")
    st.write("Welcome to ImageMagic! Choose a feature from the sidebar to get started.")
    st.image("Background_remove.png", caption="Background Removal Example", width=600)
    st.image("pencil_sketch.png", caption="Pencil Sketch Example", width=600)

elif selected_option == "Search image":
    st.title("🔍 Search Images")
    search_query = st.text_input("Enter a search term:")
    if st.button("Search"):
        if search_query:
            search_unsplash_images(search_query)
        else:
            st.warning("Please enter a search query.")

else:
    # Upload and process image
    my_upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if my_upload is not None:
        if my_upload.size > MAX_FILE_SIZE:
            st.error("File too large! Please upload an image smaller than 5MB.")
        else:
            process_image(upload=my_upload, action=selected_option)
