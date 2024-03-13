
import streamlit as st
import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import random
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input



#Streamlit webpage layout
#Apply custom CSS for background color
st.markdown(
    """
    <style>
        .stApp {
            background-color: #add8e6; /* Set the background color to light pastel blue */
        }
    </style>
    """,
    unsafe_allow_html=True
)
uwu_cat_image_path = "/mnt/c/Users/mattm/OneDrive/Documents/ecs111/ecs111_project_pet_emotions_repo/ecs111/uwu_cat.jpeg"
uwu_cat_image = Image.open(uwu_cat_image_path)

# Adjust width and position of the image
uwu_image_column, uwu_content_column = st.columns([1, 2.3])  # Create two columns, adjust width as needed
with uwu_image_column:
    st.image(uwu_cat_image, use_column_width=True)  # Adjust width as needed

# Title & description
with uwu_content_column:
    st.markdown('<link href="https://fonts.googleapis.com/css2?family=DynaPuff&display=swap" rel="stylesheet">', unsafe_allow_html=True)
    st.markdown('<h1 style="font-family: \'DynaPuff\'; font-size: 52px;">Pet Feelz 🐾</h1>', unsafe_allow_html=True)
    st.write("Welcome to Pet Feelz!")
    st.write("Here, you can upload an image of your adorable pet and receive suggestions for improving or maintaining your **pet's 'feelz'** based on the emotion that we've detected!")



model = tf.keras.models.load_model('/mnt/c/Users/mattm/OneDrive/Documents/ecs111/ecs111_project_pet_emotions_repo/ecs111/95_model.h5')
ds_name = 'Pets Facial Expression'
data_dir = 'pet_images'

def gen_data_and_labels(data_dir):
    #Identifies and stores filepaths to images in images variables
    #Labels images according to the file they were found in.
    
    image_paths = []
    labels = []

    files = os.listdir(data_dir)
    for file in files:
        
        if file == 'Master Folder':
            continue
            
        filepath = os.path.join(data_dir, file)
        
        imagelist = os.listdir(filepath)
      
        for im in imagelist:
           
            im_path = os.path.join(filepath, im)
            image_paths.append(im_path)
            labels.append(file)
            
    return image_paths, labels


image_paths, labels = gen_data_and_labels(data_dir)

def create_df(image_paths, labels):

    Fseries = pd.Series(image_paths, name= 'image_paths')
    Lseries = pd.Series(labels, name='labels')
    df = pd.concat([Fseries, Lseries], axis= 1)
    return df

df = create_df(image_paths, labels)

happy = df.groupby("labels").get_group("happy")
Sad = df.groupby("labels").get_group("Sad")
Angry = df.groupby("labels").get_group("Angry")
Other = df.groupby("labels").get_group("Other")


def plot_emotions(emotion):
  
    fig, axes = plt.subplots(ncols = 5,nrows = 1, figsize=(20, 20))
    for i in range(0,5):
        index = random.sample(range(len(emotion)),1)
        index = int(''.join(map(str, index)))
        

        filename = emotion.iloc[index]["image_paths"]
        label = emotion.iloc[index]["labels"]
        image = Image.open(filename)
        axes[i].imshow(image)
        axes[i].set_title("Labels: " + label, fontsize = 30)
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            
        
    

    plt.show()
    st.pyplot(fig)
    
 
grouped_data = df.groupby("labels")
num_images_per_category = 5

def load_image(image_file):
    img = Image.open(image_file)
    return img

def predict_emotion(img, model):
    # Resize and preprocess the image for your model
    #img = tf.keras.preprocessing.image.load_img(img, target_size=(300, 300))
    img = img.resize((300, 300)) 
    img_array =  tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    img = img.resize((300, 300))  # Adjust target size to your model's expected input size
    #img_array = np.array(img) / 255.0  # Normalize the image array
    #img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict the emotion
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    acc = np.max(prediction)
    class_indices = {'angry': 0, 'other': 1, 'sad': 2, 'happy': 3}
    class_labels = list(class_indices.keys())
    predicted_class_label = class_labels[predicted_class_index]
    #predictions = model.predict(img_array)
    #emotion_index = np.argmax(predictions)  # Assuming the model outputs class indices
    #emotions = ['Happy', 'Sad', 'Angry', 'Relaxed']  # Adjust according to your model
    return predicted_class_label, acc
    #return emotions[emotion_index]

# Streamlit webpage layout
st.title('Pet Emotion Classifier')
st.write("Upload an image of your pet to classify its emotion.")

# File uploader allows user to add their own image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg"])
class_labels = ['angry', 'other', 'sad', 'Happy']

def show_similar_img(label):
    if label == "angry":
        plot_emotions(Angry)
    elif label == "other":
        plot_emotions(Other)
    elif label == "sad":
        plot_emotions(Sad)
    elif label == "happy":
        plot_emotions(happy)

if uploaded_file is not None:
    image = load_image(uploaded_file)
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(' ')

    with col2:
        st.image(image, caption='Uploaded Image', use_column_width = True)

    with col3:
        st.write(' ')
    
    #Creating placeholder such that we can update "classifying -> classified"
    st.write("")
    classification_placeholder = st.empty()
    classification_placeholder.write("Classifying...")

    #Performing classification
    label = predict_emotion(image, model)

    #Updating placeholder post-classification
    classification_placeholder.write("Classified!")

    label, acc = predict_emotion(image, model)
    acc = "{:.2f}".format(100*acc)
    if label == 'other':
        st.write(f"With a probability of {acc}%:")
        st.markdown("""
        <div style='background-color: #daa520; padding: 10px; border-radius: 5px; border: 2px solid black;'>
            Your pet appears to be feeling <span style='font-family: "DynaPuff", sans-serif;'>CONTENT! 😌</span>
        </div>
        """, unsafe_allow_html=True)
        st.write(" ")
        st.markdown("""
        <div style='background-color: #fff8dc; padding: 10px; border-radius: 5px; border: 2px solid black;'>
            <b>🐾 We suggest continuing doing what you're doing so your pet remains comfortable!</b>
        </div>
        """, unsafe_allow_html=True)
    elif label == 'angry':
        st.write(f"With a probability of {acc}%:")
        st.markdown("""
        <div style='background-color: #daa520; padding: 10px; border-radius: 5px; border: 2px solid black;'>
            Your pet appears to be feeling <span style='font-family: "DynaPuff", sans-serif;'>ANGRY! 😠</span>
        </div>
        """, unsafe_allow_html=True)
        st.write(" ")
        st.markdown("""
        <div style='background-color: #fff8dc; padding: 10px; border-radius: 5px; border: 2px solid black;'>
            <b>🐾 We suggest giving your pet some space to calm down and maybe a treat too!</b>
        </div>
        """, unsafe_allow_html=True)
    
    elif label == 'sad':
        st.write(f"With a probability of {acc}%:")
        st.markdown("""
        <div style='background-color: #daa520; padding: 10px; border-radius: 5px; border: 2px solid black;'>
            Your pet appears to be feeling <span style='font-family: "DynaPuff", sans-serif;'>SAD! 🥺</span>
        </div>
        """, unsafe_allow_html=True)
        st.write(" ")
        st.markdown("""
        <div style='background-color: #fff8dc; padding: 10px; border-radius: 5px; border: 2px solid black;'>
            <b>🐾 We suggest providing your pet with comfort and lots of love!</b>
        </div>
        """, unsafe_allow_html=True)
    elif label == 'happy':
        st.write(f"With a probability of {acc}%:")
        st.markdown("""
        <div style='background-color: #daa520; padding: 10px; border-radius: 5px; border: 2px solid black;'>
            Your pet appears to be feeling <span style='font-family: "DynaPuff", sans-serif;'>HAPPY! 🤗</span>
        </div>
        """, unsafe_allow_html=True)
        st.write(" ")
        st.markdown("""
        <div style='background-color: #fff8dc; padding: 10px; border-radius: 5px; border: 2px solid black;'>
            <b>🐾 We suggest giving your pet some extra cuddles and playtime!</b>
        </div>
        """, unsafe_allow_html=True)
    else:
        print("")
        
    st.write(f"Other {label} pets look like this" )
    show_similar_img(label)


# Copyright
st.write(" ")
st.write("© 2024 MRL. All rights reserved.")
  

    
