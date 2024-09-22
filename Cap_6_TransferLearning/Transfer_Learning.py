# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 15:01:41 2021

@author: MBI
"""
#%% Modulo para decodificar predicciones
from tensorflow.keras.applications.vgg16 import decode_predictions,preprocess_input
#from tensorflow.keras.applications.resnet50 import decode_predictions,preprocess_input
#from tensorflow.keras.applications.inception_v3 import decode_predictions,preprocess_input
from tensorflow.keras.applications import VGG16

# Importar variable comunes
from tensorflow.keras.preprocessing import  image
import numpy as np
import  matplotlib.pyplot as plt
from os import  listdir

#%% Modelo Vgg16

img_he_v,img_we_v = 224,224

modelo_vgg16 = VGG16(
    weights='imagenet', 
    include_top=True, 
    input_shape=(img_he_v, img_we_v, 3)
    )

#%% Modelo ResNet50
"""
img_he_r,img_we_r = 299,299

modelo_resnet50 = ResNet50(
    weights='imagenet',
    include_top=True,
    input_shape=(img_he_r,img_we_r,3)
    )
"""
#%% Modelo InceptionV3
"""
img_he_in,img_we_in = 299,299

modelo_inceptionV3 = InceptionV3(
    weights='imagenet',
    include_top=True,
    input_shape=(img_he_in,img_we_in,3)
    
    )
"""

#%% Presicion del Modelo

path = "C:/Users/MBI/Documents/Python_Scripts/Tensorflow_Vision_Computacional/Cap-6_TransferLearning/Images"

imag = listdir(path=path)
fig = plt.figure(figsize=(10,8))
row,column,i = 3,3,0

for img in imag:
    i += 1
    im = image.load_img(path+"/"+img,target_size=(img_he_v,img_we_v))
    imagen = image.img_to_array(im)
    imagen = np.expand_dims(imagen,axis=0)
    imagen = preprocess_input(imagen)
    
    label_pred = modelo_vgg16.predict(imagen,verbose=False)
    #label = decode_predictions(label_pred)
    #label = label[0][0]
    
    fig.add_subplot(column,row,i)
    fig.subplots_adjust(hspace=0.5,wspace=0.5)
    
    accuracy = "%.1f"% round(np.max(label_pred)*100,1)
    plt.title("Precision de el modelo {}".format(accuracy))
    plt.imshow(im)
    plt.axis('off')
plt.show()
# Nota no se puede uar la funcion decode_predictions por no tener acceso a la pagina de tensorflow.Asumo el valor arrojado como la presicion del modelo 


