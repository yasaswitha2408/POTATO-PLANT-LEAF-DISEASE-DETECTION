import cv2
from matplotlib import pyplot as plt
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

filepath = 'C:\\Users\\Sweety Chittineni\\Downloads\\projects\\Imp_DL_potato\\potatoes.h5'
model = load_model(filepath)
print(model)

print("Model Loaded Successfully")

potato_plant = cv2.imread('C:\\Users\\Sweety Chittineni\\Downloads\\projects\\Imp_DL_potato\\Plant_village\\Potato___Early_blight\\fdc1f5ed-66b5-4564-8957-055905b8a569___RS_Early.B 8244.JPG')
test_image = cv2.resize(potato_plant, (256,256)) # load image 
  
test_image = img_to_array(test_image)/255 # convert image to np array and normalize
test_image = np.expand_dims(test_image, axis = 0) # change dimention 3D to 4D
  
result = model.predict(test_image) # predict diseased palnt or not
print(result)


pred = np.argmax(result, axis=1)
print(pred)
if pred==0:
    print( "Potato___Early_blight")
       
elif pred==1:
    print("Potato___Late_blight")
        
elif pred==2:
    print("Potato___healthy")