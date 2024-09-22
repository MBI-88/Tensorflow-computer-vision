#%% Librerias
from matplotlib import colors
import tensorflow as tf
import cv2,os
import pandas as pd 
from time import sleep
import numpy as np 
import matplotlib.pyplot as plt

# %% Procesameinto de imagenes de entrenamiento
 
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
image_size = 250

cap = cv2.VideoCapture(0)
img_count = 0

if  not os.path.exists("dataset_train"):
    os.makedirs("dataset_train")

while True:
    ret,frame = cap.read(0)
    face = face_cascade.detectMultiScale(frame,1.5,5)
    for x,y,w,h in face:
        if w > 130:
            detect_face = frame[int(y):int(y+h),int(x):int(x+w)]
            cv2.imshow("Training",detect_face)
    if not ret:
        break
    k = cv2.waitKey(1)
    if k % 256 == 27:
        # Escape
        print("Escape fue presionado")
        break
    elif k % 256 == 32:
        # Espaacio
        face_resize = cv2.resize(frame,(image_size,image_size))
        image_name = "dataset_train/trainimg_{}.jpg".format(img_count)
        cv2.imwrite(image_name,face_resize)
        print("Imagen: {} guardada".format(image_name))
        img_count += 1

cap.release()
cv2.destroyWindow("Training")
#%%  Procesameinto de imagenes de test

if  not os.path.exists("dataset_test"):
    os.makedirs("dataset_test")

cap = cv2.VideoCapture(0)
img_count = 0
while True:
    ret,frame = cap.read(0)
    face = face_cascade.detectMultiScale(frame,1.5,5)
    for x,y,w,h in face:
        if w > 130:
            detect_face = frame[int(y):int(y+h),int(x):int(x+w)]
            cv2.imshow("Test",detect_face)
    if not ret:
        break
    k = cv2.waitKey(1)
    if k % 256 == 27:
        # Escape
        print("Escape fue presionado")
        break
    elif k % 256 == 32:
        # Espaacio
        face_resize = cv2.resize(frame,(image_size,image_size))
        image_name = "dataset_test/testimg_{}.jpg".format(img_count)
        cv2.imwrite(image_name,face_resize)
        print("Imagen: {} guardada".format(image_name))
        img_count += 1

cap.release()
cv2.destroyWindow("Test")

# %% Prcesando dataset

training_path = "trainimgface.csv"
test_path = "testimgface.csv"

training_data = pd.read_csv(training_path)
test_data = pd.read_csv(test_path)

training_data.head()
# %% Procesando dataset

test_data.head()

# %% Procesando train data

coltrn = training_data["image"]
imgs = []
training = training_data.drop("image",axis=1)
Y_train = []

for i in range(coltrn.shape[0]):
    p = os.path.join(os.getcwd(),'dataset_train/'+str(coltrn.iloc[i]))
    imag  = cv2.imread(p,1)
    gray_imag = cv2.cvtColor(imag,cv2.COLOR_BGR2GRAY)
    imgs.append(gray_imag)
    y = training.iloc[i,:]
    Y_train.append(y)

X_train = np.asanyarray(imgs)
Y_train = np.array(Y_train,dtype="float")

X_train.shape,"  ",Y_train.shape
# %% Procesando test data

coltest = test_data["image"]
imgstest = []
testing = test_data.drop("image",axis=1)
Y_test = []

for x in range(coltest.shape[0]):
    path = os.path.join(os.getcwd(),'dataset_test/'+str(coltest.iloc[x]))
    img = cv2.imread(path,1)
    gray_imag = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgstest.append(gray_imag)
    y = testing.iloc[x,:]
    Y_test.append(y)

X_test = np.asanyarray(imgstest)
Y_test = np.array(Y_test,dtype="float")
X_test.shape,"  ",Y_test.shape

# %%  Procesando bloque de entrenameinto

Y_trainx = training.loc[:,['0x','1x','2x','3x','4x','5x','6x','7x','8x','9x','10x','11x','12x','13x','14x','15x']]
Y_trainy = training.loc[:,['0y','1y','2y','3y','4y','5y','6y','7y','8y','9y','10y','11y','12y','13y','14y','15y']]

Y_testx = testing.loc[:,['0x','1x','2x','3x','4x','5x','6x','7x','8x','9x','10x','11x','12x','13x','14x','15x']] 
Y_testy = testing.loc[:,['0y','1y','2y','3y','4y','5y','6y','7y','8y','9y','10y','11y','12y','13y','14y','15y']]


Y_trainx.shape,"  ",Y_trainy.shape,"  ",Y_testx.shape,"  ",Y_testy.shape
# %% Ejemplo de muestra 

x0 = Y_trainx.iloc[0,:]
y0 = Y_trainy.iloc[0,:]
plt.imshow(np.squeeze(X_train[10]),cmap="gray")
plt.scatter(x0,y0,color="red")
plt.show()


# %% Ejemplo de prueba

x0t=Y_testx.iloc[6,:]
y0t=Y_testy.iloc[6,:]
plt.imshow(np.squeeze(X_test[6]),cmap='gray')
plt.scatter(x0t, y0t,color ='red')
plt.show()

# %% Modelo 

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),input_shape=(250,250,1),padding="same",activation="relu"),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64,(3,3),activation="relu"),
    tf.keras.layers.Conv2D(64,(3,3),activation="relu"),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(128,(2,2),activation="relu"),
    tf.keras.layers.Conv2D(128,(2,2),activation="relu"),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(256,(2,2),activation="relu"),
    tf.keras.layers.Conv2D(256,(2,2),activation="relu"),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(500,activation="relu"),
    tf.keras.layers.Dense(500,activation="relu"),
    tf.keras.layers.Dense(32)
])

model.summary()
# %% Metricas del modelo

model.compile(optimizer="Adam",loss="mse",metrics=["accuracy"])

# %% Entrenamiento del modelo

X_train = X_train.reshape(50,250,250,1)
X_test = X_test.reshape(7,250,250,1) 
history = model.fit(X_train,Y_train,validation_data=(X_test,Y_test),epochs=20,batch_size=10)

# %% Visualizacion de metricas

plt.figure(figsize=(12,5))
plt.subplot(121)
plt.title("Model accuracy")
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(['Train','Test'],loc="upper left")
plt.subplot(122)
plt.title("Metric loss")
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Loss train","Loss test"],loc="upper left")
plt.show()
# %% Evaluacion del modelo

y_val = model.predict(X_test)
y_valx = y_val[::1,::2]
y_valy = y_val[:,1::2]

plt.imshow(np.squeeze(X_test[5]),cmap="gray")
plt.scatter(y_valx[5],y_valy[5],color="red")
plt.show()