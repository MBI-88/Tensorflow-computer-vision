# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 15:01:16 2021

@author: MBI
"""
#%% Librerias
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from tensorflow.keras.layers import Dense,Flatten,Dropout,GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import os
#%% Ajueste de datos

img_height,img_width = 224,224
num_epochs = 5
batch_size = 10
num_class = 5
class_dict = {0:"Daisy",1:"Dandelion",2:"Rose",3:"Sunflower",4:"Tulip"}

train_path = "C:/Users/MBI/Documents/Python_Scripts/Tensorflow_Vision_Computacional/Cap-5_Arquitecture/flowers_train"
validation_path = "C:/Users/MBI/Documents/Python_Scripts/Tensorflow_Vision_Computacional/Cap-5_Arquitecture/flowers_test"

train_gene = ImageDataGenerator(
    preprocessing_function = preprocess_input,
    rotation_range = 90,
    horizontal_flip = True,
    vertical_flip = True,
    )

validation_gene = ImageDataGenerator(
    preprocessing_function = preprocess_input,
    rotation_range = 90,
    horizontal_flip = True,
    vertical_flip = True,
    )

train_data = train_gene.flow_from_directory(train_path,target_size=(img_height,img_width),batch_size=batch_size,seed=13,shuffle=True)
validation_data = validation_gene.flow_from_directory(validation_path,target_size=(img_height,img_width),batch_size=batch_size,seed=13,shuffle=True)

train_rose = os.path.join(train_path,'rose')
train_daisy = os.path.join(train_path,'daisy')
train_dande =os.path.join(train_path,'dandelion')
train_sunf = os.path.join(train_path,'sunflower')
train_tu = os.path.join(train_path,'tulip')

test_rose = os.path.join(validation_path, 'rose')
test_sunf = os.path.join(validation_path, 'sunflower')
test_tu = os.path.join(validation_path, 'tulip')
test_dande = os.path.join(validation_path, 'dandelion')
test_daisy = os.path.join(validation_path, 'daisy')

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
num_total_val = num_test_daisy+num_test_dan+num_test_rose+num_test_sun+num_test_tu

print(num_total_train," ",num_total_val)

#%% Cargando el modelo RestNet50

modelo_resnet50 = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(img_height,img_width,3)
    )

#%% Construccion del Transfer Learning

def transfer_RestNet50(base_model,layers,dropout,num_class):
    for layer in base_model.layers:
        layer.trainable = False
    
    X = base_model.output 
    X = GlobalAveragePooling2D()(X)
    X = Flatten()(X)
    
    for fc in layers:
        X = Dense(fc,activation='relu')(X)
        X = Dropout(dropout)(X)
    
    prediction = Dense(num_class,activation='softmax')(X)
    
    final_model = Model(inputs=base_model.inputs,outputs=prediction)
    return final_model


#%% Compilacion del modelo

fc_layers = [1024, 1024]
modelo_trans = transfer_RestNet50(modelo_resnet50,fc_layers,0.3,num_class)

modelo_trans.summary()
modelo_trans.compile(
    optimizer=Adam(lr=0.00001),
    loss='categorical_crossentropy',
    metrics=["accuracy"]
    )



#%% Entrenameinto del modelo

file_paht_model = "C:/Users/MBI/Documents/Python_Scripts/Tensorflow_Vision_Computacional/Cap-6_TransferLearning/model_weight.hdf5"
checkpoint = ModelCheckpoint(file_paht_model,verbose=1,save_best_only=True,monitor='loss')

history_model = modelo_trans.fit(train_data,
                                 epochs=num_epochs,
                                 steps_per_epoch=num_total_train//batch_size,
                                 validation_data=validation_data,
                                 validation_steps=num_total_val//batch_size,
                                 callbacks=[checkpoint]
                                 )



#%% Plotiando el modelo

acc = history_model.history['accuracy']
val_acc = history_model.history['val_accuracy']
loss = history_model.history['loss']
val_loss = history_model.history['val_loss']

plt.figure(figsize=(12,8))
plt.subplot(2,1,1)
plt.plot(acc,label='Training Accuracy')
plt.plot(val_acc,label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim(),1)])
plt.title("Training and Validation Accuracy")
plt.subplot(2,1,2)
plt.plot(loss,label='Training Loss')
plt.plot(val_acc,label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,5.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
