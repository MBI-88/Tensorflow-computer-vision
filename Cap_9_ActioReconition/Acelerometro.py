"""
Created on July 26 15:02:20 2021
@author: MBI
Descripcion: Scripts para determinar la posicion humana por medio de la lectura de un
acelerometro.

"""
#%%  Modulos

from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras import Sequential
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import preprocessing,metrics
import seaborn as sns
import pandas as pd
import numpy as np

#%% Analisis de datos 1
ruta = "Tensorflow_Vision_Computacional/Cap_9_ActioReconition/sample.csv"

def  conver_to_float(x):
    try:
        return np.float16(x)
    except:
        return np.nan

def read_data(ruta):
    columns = ['user-id','activity','timestamp','x-axis','y-axis','z-axis']
    data = pd.read_csv(ruta,names=columns,header=None)
    data['z-axis'].replace(regex=True,inplace=True,to_replace=r';',value=r'')
    data['z-axis'] = data['z-axis'].apply(conver_to_float)
    data.dropna(axis=0,inplace=True,how='any')
    return data

dataset = read_data(ruta)
dataset.head(10)

#%% Analisis de datos 2

dataset_test = dataset[dataset['user-id'] > 18] # matriz de datos
dataset_train = dataset[dataset['user-id'] <= 18] # matriz de datos

target_test = dataset_test.pop('activity') #-> X Vector
target_train = dataset_train.pop('activity') #-> Y Vector

XData_test = np.array(dataset_test.values) # matriz de datos array
YData_test = np.array(target_test.values) # vector de datos array

XData_train = np.array(dataset_train.values) # matriz de datos array
YData_train = np.array(target_train.values) # vector de datos array

scaling_train = preprocessing.MinMaxScaler(feature_range=(0,1))
XData_train = scaling_train.fit_transform(XData_train)

scaling_test = preprocessing.MinMaxScaler(feature_range=(0,1))
XData_test = scaling_train.fit_transform(XData_test)

dataset_tf_train = tf.data.Dataset.from_tensor_slices((XData_train,YData_train)) # dataset de tensores
dataset_tf_test = tf.data.Dataset.from_tensor_slices((XData_test,YData_test)) # dataset de tensores
#%% Analisis de datos 3

train_ds = dataset_tf_train.shuffle(len(dataset_train)).batch(1)
test_ds = dataset_tf_test.shuffle(len(dataset_test)).batch(1)
#%% Creacion del modelo

def model(activation):
    clf = Sequential([
        Dense(units=128,activation='relu',name='dense_1'),
        Dense(units=128,activation='relu',name='dense_2'),
        Dense(units=128,activation='relu',name='dense_3'),
        Flatten(),
        Dense(units=6,activation=activation,name='dense_4')
    ])
    clf.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return (clf,activation)

modelo_sig,activation = model('sigmoid')
#%% Entrenamiento sigmoid

history = modelo_sig.fit(train_ds,validation_data=test_ds,epochs=5)
#%% Graficas Sigmoid

def  show_history(history,activation):
    plt.figure(figsize=(12,8))
    plt.suptitle("Model using "+activation)
    plt.subplot(121)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title("Model loss")
    plt.xlabel("Epoch")
    plt.ylabel('Magnitude')
    plt.legend(['Train','Test'],loc='upper left')
    plt.subplot(122)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title("Model metrics")
    plt.xlabel('Epoch')
    plt.ylabel('Magnitude')
    plt.legend(['Train','Test'],loc='upper left')
    plt.show()

show_history(history,activation)

#%% Graficos Softmax/entrenamiento

modelo_soft,activation = model('softmax')

history = modelo_soft.fit(train_ds,validation_data=test_ds,epochs=5)

show_history(history,activation)

#%% Validando  la precision del modelo sigmoid

Labels = ['Jogging',
          'Walking',
          'Upstairs',
          'Downstairs',
          'Sitting',
          'Standing']

def show_matriz_cofution(test,prediction):
    cn = metrics.confusion_matrix(test,prediction)
    plt.figure(figsize=(6,8))
    sns.heatmap(
        cn,
        cmap='winter',
        linecolor='yellow',
        linewidths=2,
        xticklabels=Labels,
        yticklabels=Labels,
        annot=True,
        fmt='d')
    plt.title("Heat map Confution Matrix "+activation)
    plt.ylabel("Acutal value")
    plt.xlabel('Predicted values')
    plt.show()

predictions = modelo_sig.predict_classes(XData_test)

y_pred_max = tf.argmax(predictions,axis=None,output_type=tf.int32)
y_test_max = tf.argmax(YData_test,axis=None,output_type=tf.int32)

print("Max value y_pred:{}  Max value y_test: {}".format(y_pred_max,y_test_max))

show_matriz_cofution(YData_test,predictions)
#%% Validando presicion del modelo softmax

predictions = modelo_soft.predict_classes(XData_test)

y_pred_max = tf.argmax(predictions,axis=None,output_type=tf.int32)
y_test_max = tf.argmax(YData_test,axis=None,output_type=tf.int32)

print("Max value y_pred:{}  Max value y_test: {}".format(y_pred_max,y_test_max))

show_matriz_cofution(YData_test,predictions)









