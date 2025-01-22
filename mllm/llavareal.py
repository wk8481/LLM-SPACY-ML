import streamlit as st
import requests
import json
import base64
from PIL import Image
from io import BytesIO

# Function to process the image and get the response
def get_llava_response(image_url, prompt):
    # Read image from URL and encode it in base64
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    encoded_string = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # API endpoint and payload
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llava",
        "prompt": prompt,
        "stream": False,
        "images": [encoded_string]
    }

    # Send POST request
    response = requests.post(url, data=json.dumps(payload), headers={"Content-Type": "application/json"})

    # Return the response
    return response.text

# Streamlit UI
st.title("LLaVA Image Analysis")
st.write("Provide an image URL and a prompt for analysis.")

# User input for image URL
image_url = st.text_input("Image URL: ", "")

# User input for prompt
prompt = st.text_input("Prompt: ", "What is in this picture?")

# Generate response when user submits input
if st.button("Analyze") and image_url:
    response = get_llava_response(image_url, prompt)
    st.write("Response from LLaVA:")
    st.write(response)