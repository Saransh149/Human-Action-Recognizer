import streamlit as st
import requests
import io
from PIL import Image

def main():
    st.title("Human Action Recognizer")

    # Input field for user to upload image
    user_image = st.file_uploader("Upload an image", type=["jpg", "jpeg"])

    # Button to submit the image
    if st.button("Submit"):
        if user_image is not None:
            # Call the FastAPI endpoint and get predictions
            predictions = get_predictions(user_image)

            # Display the predictions
            st.write("Predictions:")
            st.write(predictions)
        else:
            st.warning("Please upload an image first.")

# Function to call the FastAPI endpoint and get predictions
def get_predictions(image):
    api_url = "http://localhost:8000/predict" 
    files = {"file": (image.name, image, image.type)}
    response = requests.post(api_url, files=files)
    prediction = response.json()
    return prediction

if __name__ == "__main__":
    main()
