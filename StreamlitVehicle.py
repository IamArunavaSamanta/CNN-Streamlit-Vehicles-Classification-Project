import streamlit as st
import time as t
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix, classification_report
import keras
from keras.activations import relu, linear
from keras.layers import Dense, Dropout,Input
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping,ReduceLROnPlateau, TensorBoard
from keras.regularizers import L1, L2
from keras.layers import Conv2D, MaxPool2D, Flatten
import cv2, os
import datetime
import joblib

st.set_page_config(
    page_title="Arunava's Streamlit",
    page_icon="𓆩♛𓆪"
)
st.write('# :rainbow[Vehicle Classification]🔥')
st.info("Vehicle classification is a computer vision task that uses machine learning to automatically identify and categorize vehicles\
             like cars, trucks,  bikes from images.")  

# Inject custom CSS for gradient sidebar
st.markdown("""
    <style>
        [data-testid="stSidebar"] > div:first-child {
            background-image: linear-gradient(to bottom right, #3F0112, #0E1117);
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

st.sidebar.image("https://github.com/IamArunavaSamanta/CNN-Streamlit-Vehicles-Classification-Project/blob/main/images/logo.png?raw=true", width=100)
menu = st.sidebar.radio("", ["🏠Home", "🔐Login Page", "💻Prediction", "📍Fun Quiz", "⭐Feedback"])
#--------------------------------------------------------------------------------------------------------------------------

if menu == "🏠Home":
    st.image("https://github.com/IamArunavaSamanta/CNN-Streamlit-Vehicles-Classification-Project/blob/main/images/Vehicles1.png?raw=true", width=600)
    st.markdown('''#### :red-background[:orange[CNN]]:orange[, or Convolutional Neural Network,] ''')
    
    st.markdown("""
    It is a powerful class of deep learning models designed to process and classify visual data. In the context of vehicle image classification, CNNs can learn to identify features such as shapes, edges, and textures that distinguish different types of vehicles.
    
    ##### 🔍 How CNN Works:
    - **Convolution Layers**: Extract features from the image using filters.
    - **Pooling Layers**: Reduce the spatial dimensions, making the model efficient.
    - **Fully Connected Layers**: Interpret the extracted features to classify the image.
    - **Activation Functions**: Introduce non-linearity to learn complex patterns.
    
    ##### 🚘 Applications:
    - Classifying vehicle types (e.g., car, truck, bike)
    - Traffic monitoring systems
    - Autonomous driving
    - Smart parking solutions
    
    CNNs are trained on labeled datasets and can achieve high accuracy in recognizing vehicle types even under varying conditions like lighting, angle, and background.
    """)
        
    st.markdown(
        "---🔗Developed by [Arunava Samanta](https://www.linkedin.com/in/arunava-samanta-7439071ba/)"
    )

#--------------------------------------------------------------------------------------------------------------------
if menu == "🔐Login Page":
    st.markdown("""
    <h5><span style='color: orange; font-size: 36px;'>Login</span></h5>
    """, unsafe_allow_html=True)
    email = st.text_input("Email Id")
    pw = st.text_input("Password", type='password')
    if st.button("Submit"):
        if email == 'arunava' and pw == '1234':
            st.success("Logged in successfully")
        else: 
            st.error("Wrong password. Try again")
#-------------------------------------------------------------------------------------------------------------------
if menu =='💻Prediction':
    st.markdown("##### Upload a vehicle image 👇")
    uploaded_file = st.file_uploader("Choose a JPG file", type=["jpg"])

        #Function
    class ModelWrapper():
        def __init__(self, model, encoder):
            self.model = model
            self.encoder = encoder
            
        def single_img_read(self, file_obj):
            from PIL import Image
            IMAGE_INPUT_SIZE = 200
        
            # Read image from file-like object using PIL, then convert to NumPy array
            img = Image.open(file_obj).convert("L")  # Convert to grayscale
            img = img.resize((IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE))
            img = np.array(img)
            # Reshape for model input
            img = img.reshape(1, IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE, 1)
            # Predict
            z = self.model.predict(img)
            index = np.argmax(z)
            Predicted_accuracy = z[0][index]
            predicted_label = self.encoder.inverse_transform([index])
            
            # Bar chart of probabilities
            category_labels = self.encoder.classes_  # Assuming encoder has all class labels
            percentages = z[0] * 100
            fig, ax = plt.subplots(figsize=(7, 2))
            fig.patch.set_facecolor((0, 0, 0, 0.6))  #Set figure background to black
            ax.set_facecolor((0, 0, 0, 0.2))  
            # Define 7 distinct colors
            c = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta']
            ax.bar(category_labels, percentages, color=c, edgecolor='white')
            ax.set_xlabel('Categories', color='white')
            ax.set_ylabel('Probability (%)', color='white')
            ax.set_ylim(0, 100)
            plt.xticks(rotation=35, color='white')
            
            # Set tick labels to white
            ax.tick_params(axis='y', colors='white')
            # Set axis lines (spines) to white
            for spine in ax.spines.values():
                spine.set_color('white')
            st.pyplot(fig)
            
            return Predicted_accuracy, predicted_label

    model = joblib.load("models/VehicleModel.pkl")
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", width=300)
    
        # Run prediction only after file is uploaded
        accuracy, label = savedModel.single_img_read(uploaded_file)
    
        st.markdown(f"### 🤖 Predicted Class: `{label[0]}`")
        st.markdown(f"### 📊 Model Accuracy: `{accuracy:.3f}`")
        st.success("✅ Prediction completed!")
    else:
        st.info("Please upload a vehicle image to see predictions.")
#--------------------------------------------------------------------------------------------------------        
    st.text("🗒 Examples")    
    # Image paths or URLs
    images = ["https://github.com/IamArunavaSamanta/CNN-Streamlit-Vehicles-Classification-Project/blob/main/images/Bike.jpg?raw=true", 
              "https://github.com/IamArunavaSamanta/CNN-Streamlit-Vehicles-Classification-Project/blob/main/images/Car.jpg?raw=true",
              "https://github.com/IamArunavaSamanta/CNN-Streamlit-Vehicles-Classification-Project/blob/main/images/Plane.jpg?raw=true"]
    
    # Create 3 columns
    col1, col2, col3 = st.columns(3)
    
    # Set a fixed width (e.g., 200px)
    image_width = 200
    
    # Display images
    col1.image(images[0], width=image_width)
    col1.success("🛵 Predicted Class: Bikes")
    col1.success("📊 Model Accuracy: 96.1%")
    col2.image(images[1], width=image_width)
    col2.success("🚗 Predicted Class: Cars")
    col2.success("📊 Model Accuracy: 90.1%")
    col3.image(images[2], width=image_width)
    col3.success("✈️ Predicted Class: Planes")
    col3.success("📊 Model Accuracy: 86.8%")

#---------------------------------------------------------------------------------------------------
if menu == '📍Fun Quiz':
    st.text("1. Which of the following is a type of Machine Learning?")
    ans = st.radio("Choose any one",  ['A. Supervised', 'B. Unsupervised', 'C. Reinforcement', 'D. All of the above'], index=None)
    if ans is None:
        st.warning("⚠️ Select an option")
    elif ans == 'D. All of the above':
        st.success("✅ Correct!")
    else:
        st.error("❌ Try again")
#---------------------------------------------------------------------------------------------------        
    st.text("2. Which component of a neural network adjusts weights during training?")
    ans = st.radio("Choose any one",  ['A. Loss function', 'B. Optimizer', 'C. Activation Function', 'D. Bias'], index=None)
    if ans is None:
        st.warning("⚠️ Select an option")
    elif ans == 'B. Optimizer':
        st.success("✅ Correct!")
    else:
        st.error("❌ Try again")
#------------------------------------------------------------------------------------------------------        
    st.text("3.  What is the full name of the OpenCV?")
    ans = st.radio("Choose any one",  ['A. Open Computer Vector', 'B. Open Computer Vision', 'C. Open Common Vector', 'D.  Open Common Vision'], index=None)
    if ans is None:
        st.warning("⚠️ Select an option")
    elif ans == 'B. Open Computer Vision':
        st.success("✅ Correct!")
    else:
        st.error("❌ Try again")
    if st.button("Submit"):
        st.balloons()
#-------------------------------------------------------------------------------------------------------------
if menu == "⭐Feedback":
    st.markdown("""
        <style>  
            .stats-number {
                font-size: 1.5rem;
                font-weight: 500;
                color: #FFD700;
                margin-bottom: 0.5rem;
            }
            
            /* Feature icons */
            .feature-icon {
                font-size: 2rem;
                margin-bottom: 1rem;
            }
            
            /* Stats cards */
            .stats-card {
                background: linear-gradient(to bottom right, #3F0112, #0E1117);
                border-radius: 20px;
                padding: 1.5rem;
                text-align: center;
                margin: 0.5rem;
            }
        </style>
    """, unsafe_allow_html=True)
#---------------------------------------------------------------------------------------------------------------------    
    st.markdown("##### 📈 Model Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
            <div class='stats-card'>
                <div class='feature-icon'>🎯</div>
                <div class='stats-number'>86.1%</div>
                <div class='stats-label'>Accuracy</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='stats-card'>
                <div class='feature-icon'>⚡</div>
                <div class='stats-number'>2.3s</div>
                <div class='stats-label'>Analysis Time</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div class='stats-card'>
                <div class='feature-icon'>✉️</div>
                <div class='stats-number'>5400+</div>
                <div class='stats-label'>Trained Img</div>
            </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
            <div class='stats-card'>
                <div class='feature-icon'>🔎</div>
                <div class='stats-number'>7</div>
                <div class='stats-label'>Total Class</div>
            </div>
        """, unsafe_allow_html=True)
#--------------------------------------------------------------------------------------------------------------    
    st.text("")
    st.markdown("##### ⭐ Feedback")
    rating = st.radio('Rating',['Very Bad', 'Bad', 'Average', 'Good', 'Very Good', 'Excellent'], index=None, horizontal=True)
    feedback_comment = st.text_area("Additional comments (optional):")
    if st.button("Submit Feedback"):
        if rating is not None:
            st.success("Thank you for your feedback! 🙏")
        else:
            st.warning("Please select rating! 😊")

