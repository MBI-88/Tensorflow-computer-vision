# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 15:29:55 2021

@author: MBI
"""
#%% Teoria Previa
"""
La arquitectura de busqueda visual

Los modelos como ResNet,VGG16 y Inception pueden ser escencialmente separados en 2 componentes:
    
    .Los primeros componentes identifican los contenidos de bajo nivel de una imagen tales como 
    vectores de variables (bordes).
    .Los segundos componentes representan los contenidos de alto nivel de una  imagen, tales como caract.
    de la imagen final, los cuales estan en el ensamblado de varios contenidos de bajos nivel.


El aplanado de una imagen (flatten) sigue el siguiente algoritmo:
    .Forma de imagen de entrada (# imagenes,ancho,alto, # canales) cuando se hace el aplanado ocurre lo siguiente
     (# imagenes,ancho*alto*# canales) lo que genera un vector n-dimensional.
     
     .Ejemplo: # imagens = 1000 , ancho = 14 , alto = 14 , # canales = 512 
     resultado  final (1000,100352)


Pasos para usar el transfer-learning (transferencia de aprendisaje)

1. Para usar el transfer-learning para el desarrollo de un nuevo modelo por el desmarcado de las ultimas capas
es bien conocido para modelos tales como ResNet,VGG16 o Inception y entonces adicionar una capa personalizada
incluyendo la capa completamente conectada , dropout,activacion y softmax

2. Entrenar el nuevo modelo con nuestro dataset.
3. Subir una imagen y encontrar su vector de variables y su clase asociada , corriendo las imagenes a traves del 
nuevo modelo creado.
4. Para guardar tiempo en la busqueda, buscamos solamente dentro del directorio de clases que corresponde con la 
imagen subida.
5. Los algortmos de busquedas tales como la distancia Euclidiana o la similitud coseno son usados.
6. Mostramos los resultados si la similitud coseno es > 0.999, si no , reentrenamos el modelo con la imagen o ajustamos
los parameteros y volvemos a correr el proceso.
7. Para aumentar en velocidad , detectar la ubicacion de objetos dentro  de la imagen subida y dentro de la base de datos
de imagen de busqueda y directorio usando las bounding box generadas.



"""
#%% Modulos

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist
#%% Cargando Modelo 

modelo_base = ResNet50(
    weights='imagenet',
    include_top = False,
    input_shape = (224,224,3)
    )

path_model = "C:/Users/MBI/Documents/Python_Scripts/Tensorflow_Vision_Computacional/Cap-6_TransferLearning/model_weight.hdf5"
modelo_final = load_model(path_model)

#%% Preparacion  de datos

upload_image = "C:/Users/MBI/Documents/Python_Scripts/Tensorflow_Vision_Computacional/Cap-5_Arquitecture/flowers_train/sunflower/24459750_eb49f6e4cb_m.jpg"

def decode_image(path=''):
    if path == '':
        return print("Cant find directory")
    else:
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image,channels=3)
        image = tf.image.convert_image_dtype(image,tf.float32)
        image = tf.image.resize(image,(224,224))
        image = tf.expand_dims(image,axis=0)
        return image

image_test = decode_image(upload_image) # para modelo final
image_prepros = preprocess_input(image_test) # para modelo base

base_precdic = modelo_base.predict(image_prepros) # Modelo Base Resnet
final_predic = modelo_final.predict(image_test,verbose=False) # Modelo Final adaptado

base_feature = np.array(base_precdic)
base_flatten = base_feature.flatten()

final_feature = np.array(final_predic)
final_flatten = final_feature.flatten()

print(final_flatten)
y_prob = modelo_final.predict(image_test)
y_class = y_prob.argmax(axis=-1)
print(y_class)


#%% Filtrando el directorio

diccionario = {
   0  : "C:/Users/MBI/Documents/Python_Scripts/Tensorflow_Vision_Computacional/Cap-5_Arquitecture/flowers_train/daisy",
   1  : "C:/Users/MBI/Documents/Python_Scripts/Tensorflow_Vision_Computacional/Cap-5_Arquitecture/flowers_train/dandelion",
   2  : "C:/Users/MBI/Documents/Python_Scripts/Tensorflow_Vision_Computacional/Cap-5_Arquitecture/flowers_train/rose",
   3  : "C:/Users/MBI/Documents/Python_Scripts/Tensorflow_Vision_Computacional/Cap-5_Arquitecture/flowers_train/sunflower",
   4  : "C:/Users/MBI/Documents/Python_Scripts/Tensorflow_Vision_Computacional/Cap-5_Arquitecture/flowers_train/tulip"
    
    }

path = diccionario[y_class[-1]]

mindis = 10000 
maxcosin = 0

for label in os.listdir(path):
    image_path = os.path.join(path,label)
    
    image_sample = decode_image(image_path)
    image_sample_base = preprocess_input(image_sample)
    
    base_predic = modelo_base.predict(image_sample_base)
    base_feature = np.array(base_predic)
    base_flatten1d = base_feature.flatten()
    
    Euclidiana = dist.euclidean(base_flatten,base_flatten1d)
    
    if mindis > Euclidiana:
        mindis = Euclidiana
        minfile = label
    
    dot_product = np.dot(base_flatten,base_flatten1d)
    nor_Y = np.linalg.norm(base_flatten)
    nor_X = np.linalg.norm(base_flatten1d)
    cosin_s = dot_product / (nor_X * nor_Y)
    
    if maxcosin < cosin_s:
        maxcosin = cosin_s
        cosfile = label
    
    print("\n{} filename {} euclidian {} cosin_similitary".format(label,Euclidiana,cosin_s))
    print("\n{} minfilename {} mindis {} maxcosin {} cosfile".format(minfile,mindis,maxcosin,cosfile))
    

#%% Ploteando resultados


plt.figure(figsize=(10,8))
plt.subplot(131)
image_select = image.load_img(upload_image,target_size=(224,224))
plt.title("Image select")
plt.imshow(image_select)
plt.axis('off')
plt.subplot(132)
image_result1 = os.path.join(path,minfile)
image_result1 = image.load_img(image_result1,target_size=(224,224))
plt.imshow(image_result1)
plt.axis('off')
ecu = "%.7f"% mindis
plt.title("Eucilidian_Distance: {}".format(ecu))
plt.subplot(133)
image_result2 = os.path.join(path,cosfile)
image_result2 = image.load_img(image_result2,target_size=(224,224))
plt.imshow(image_result2)
plt.axis('off')
cosi = "%.7f"% maxcosin
plt.title("Cosine_Similitary: {}".format(cosi))

# Nota la carencia de presicion del modelo  es porque no se entreno el modelo base
