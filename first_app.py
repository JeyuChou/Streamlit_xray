#Streamlit app for xray classification

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
import matplotlib.pyplot as plt


xray_model = load_model('xray_model.h5')

st.title('xray classification app')

"""
x_ray_url= 'https://www.radiologymasterclass.co.uk/images/quizzes/Quiz-Images-Chest-1/question_1.jpg'
path = tf.keras.utils.get_file('xray',x_ray_url)

img = keras.preprocessing.image.load_img(path,color_mode='grayscale',target_size=(128,128))
img_array = keras.preprocessing.image.img_to_array(img)
img_array = img_array/255.
expanded = tf.expand_dims(img_array,0)

prediction = xray_model.predict(expanded)"""

xray_url = st.text_area('Change this to an image link for xray classification')
path = tf.keras.utils.get_file('xray',xray_url)
img = tf.keras.preprocessing.image.load_img(path,color_mode='grayscale',target_size=(128,128))

plt.title('Xray Image passed in')
plt.imshow(img)
plt.show()

img_array = img_to_array(img)
img_array = img_array/255.
expanded = tf.expand_dims(img_array,0)

prediction = xray_model.predict(expanded)
score = tf.math.sigmoid(prediction)
classes = ['NORMAL','PNEUMONIA']
txt = 'this image most likely belongs to ' + str(classes[np.argmax(score)]) + ' with ' + str(np.max(score)*100) + ' percent confidence'
st.title(txt)