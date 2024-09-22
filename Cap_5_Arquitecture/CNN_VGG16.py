# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 09:33:38 2021

@author: MBI
"""
#%% Librerias
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16,preprocess_input
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.random.set_seed(42)
#%% Preprocesamiento de datos
batch = 10
image_size = (64,64)

path_train = "Tensorflow-Vision-Computacional/Cap4-DeepLearning_on_Images/flowers_train"
path_validation = "Tensorflow-Vision-Computacional/Cap4-DeepLearning_on_Images/flowers_test"

train_generator = ImageDataGenerator(
        preprocessing_function = preprocess_input,
        rotation_range = 90,
        horizontal_flip = True,
        vertical_flip = True
    )

validator_generator = ImageDataGenerator(
    preprocessing_function = preprocess_input,
    rotation_range = 90,
    horizontal_flip = True,
    vertical_flip = True
    )


validation_data = validator_generator.flow_from_directory(path_train,target_size=image_size,batch_size=batch)
train_data = train_generator.flow_from_directory(path_validation,target_size=image_size,batch_size=batch)

train_rose = os.path.join(path_train,'rose')
train_daisy = os.path.join(path_train,'daisy')
train_dande =os.path.join(path_train,'dandelion')
train_sunf = os.path.join(path_train,'sunflower')
train_tu = os.path.join(path_train,'tulip')

test_rose = os.path.join(path_validation, 'rose')
test_sunf = os.path.join(path_validation, 'sunflower')
test_tu = os.path.join(path_validation, 'tulip')
test_dande = os.path.join(path_validation, 'dandelion')
test_daisy = os.path.join(path_validation, 'daisy')

num_train_rose = len(os.listdir(train_rose))
num_train_dan = len(os.listdir(train_dande))
num_train_sun = len(os.listdir(train_sunf))
num_train_tu = len(os.listdir(train_tu))
num_train_daisy = len(os.listdir(train_daisy))

num_test_sun = len(os.listdir(test_sunf))
num_test_rose = len(os.listdir(test_rose))
num_test_daisy = len(os.listdir(test_daisy))
num_test_dan = len(os.listdir(test_dande))
num_test_tu = len(os.listdir(test_tu))

num_total_train = num_train_daisy+num_train_dan+num_train_rose+num_train_sun+num_train_tu
num_total_test = num_test_daisy+num_test_dan+num_test_rose+num_test_sun+num_test_tu

print(num_total_train," ",num_total_test)



#%% Creacion del Modelo

base_model = VGG16(weights = 'imagenet',include_top = False,input_shape =(64,64,3))

print("Number of leyers: ",len(base_model.layers))

def build_final_model(base,fc_layers,dropout,num_class):
    for layer in base.layers:
        layer.trainable = True
    
    x = base.output 
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    
    fine_tune = 100 # Limite para parar el entrenamiento de las capas
    
    for layer in base.layers[:fine_tune]:
        layer.trainable = False
    
    for fc in fc_layers:
        x = tf.keras.layers.Dense(fc,activation='relu')(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    
    predictions = tf.keras.layers.Dense(num_class,activation='softmax')(x)
    
    model = tf.keras.Model(inputs=base.inputs,outputs=predictions)
    
    return model


fc_layers = [1024,1024]
dropout = 0.3
num_class = 5

modelo = build_final_model(base_model,fc_layers,dropout,num_class)

adam = tf.keras.optimizers.Adam(lr = 0.0001)
modelo.compile(
    optimizer =adam,
    loss='categorical_crossentropy',
    metrics = ['accuracy']
    )
        
    
modelo.summary()


#%% Entrenamieto del modelo

history = modelo.fit(train_data,steps_per_epoch=num_total_train//batch,validation_data=validation_data,validation_steps=num_total_test//batch,epochs=25)

#%% Pruba del modelo

image_path = "Tensorflow-Vision-Computacional/Cap4-DeepLearning_on_Images/flowers_test/tulip/112334842_3ecf7585dd.jpg"

img = image.load_img(image_path,target_size=image_size)
img = image.img_to_array(img)
img = tf.expand_dims(img,axis=0)
img_tensor = preprocess_input(img)

feature_map = modelo.predict(img_tensor)
plt.figure(figsize = (10,8))
plt.subplot(121)
plt.title("Mapa de variable")
plt.imshow(feature_map)
plt.axis('off')
plt.subplot(122)
plt.title("Image")
plt.imshow(img_tensor[0])
plt.axis('off')
plt.show()


#%% Visualizando capas

layers_outputs = [layer.output for layer in modelo.layers[:len(modelo.layers)]]

layers_confi = tf.keras.Model(inputs=modelo.input,outputs=layers_outputs)
activation_layers = layers_confi.predict(img_tensor)


for i in range(0,len(modelo.layers)-7):
    curren_layer = activation_layers[i]
    ns = curren_layer.shape[-1] # canales del mapa de variables (kernel)
    figure = plt.figure(figsize=(12,9))
    #ax1 = figure.add_subplot(131)
    plt.suptitle("Capa_{}\n No filters {}".format(modelo.layers[i],ns),fontsize=25,color='blue')
    plt.subplot(131)
    plt.imshow(curren_layer[0,:,:,0],cmap='viridis')
    plt.axis('off')
    
    #ax2 = figure.add_subplot(132)
    plt.subplot(132)
    plt.imshow(curren_layer[0,:,:,int(ns/2)],cmap='viridis')
    plt.axis('off')
    #ax3 = figure.add_subplot(133)
    plt.subplot(133)
    plt.imshow(curren_layer[0,:,:,ns-1],cmap='viridis')
    plt.axis('off')

#%%



