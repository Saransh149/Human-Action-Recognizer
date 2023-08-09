# Human-Action-Recognizer

**Description:**
The project aims to build a system that can analyze images of humans and predict the action they are performing. The system will take an image as input and output the action label, such as "sitting," "running," "eating," etc. The model will be trained on a dataset of human action images using deep learning techniques.

**Input:**
The input to the project is an image containing a human performing an action. This image is provided by the user either through the Streamlit UI or by sending a POST request to the Django API.

**Output:**
The output of the project is the predicted action label based on the input image. This label represents the action being performed by the human in the image, such as "sitting," "running," "eating," etc.

**Libraries and Tools Required:**

1. **Python:** The entire project is built using the Python programming language.

2. **Pandas and NumPy:** For data manipulation and preprocessing.

3. **TensorFlow and Keras:** For building and training the deep learning model.

4. **Scikit-learn:** For data preprocessing and model evaluation.

5. **Django:** To create the API that serves the trained model.

6. **Django REST framework:** For building a RESTful API using Django.

7. **Streamlit:** For creating the user interface to interact with the model.

8. **Pillow:** For image manipulation and processing.

9. **Postman:** To test the API endpoints during development.

10. **Gunicorn or uWSGI:** To deploy the Django app on a web server.

11. **Kaggle Dataset:** The dataset containing human action images for training and validation.

**Steps and Flow:**

1. User uploads an image through the Streamlit UI or sends a POST request to the Django API.
2. The Streamlit app or Django API preprocesses the image.
3. The preprocessed image is fed into the trained deep learning model.
4. The model predicts the human action label based on the input image.
5. The predicted label is returned to the user via the Streamlit UI or the API response.
6. The Streamlit app displays the predicted label along with the uploaded image.
7. The Django API responds with the predicted label to the POST request.

**Deployment:**

1. Deploy the Django API using Gunicorn or uWSGI on a web server.
2. Deploy the Streamlit UI on a separate endpoint or domain, such as using Streamlit Sharing or a cloud platform.
3. Users can access the Streamlit UI to interact with the model and get action predictions.

**Summary:**

This end-to-end project uses a combination of Python libraries and tools to create a Human Action Recognition system. It involves data preprocessing, model training, API creation using Django, user interface development with Streamlit, and deployment to make the system accessible to users. The project allows users to upload images and receive predictions for the actions performed by humans in those images.
