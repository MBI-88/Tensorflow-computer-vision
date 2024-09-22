#%% Librerias
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import  image
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import preprocess_input
from  tensorflow.keras.datasets import  cifar10
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.random.set_seed(42)

# %% Cargando datos
(X_train,y_train),(X_test,y_test) = cifar10.load_data()

print(X_train.shape," ",y_train.shape," ",X_test.shape," ",y_test.shape)
# %% Procesando datos
clases = ["airplane",'automobile','bird','cat','deer','dog','frog','horse','ship','trunk']

train_set = ImageDataGenerator(
    preprocessing_function = preprocess_input, 
    rotation_range = 90,
    horizontal_flip = True, 
    vertical_flip = True,   

)
validation_set = ImageDataGenerator(
    preprocessing_function = preprocess_input, 
    rotation_range = 90,
    horizontal_flip = True, 
    vertical_flip = True,   
    )

train_set.fit(X_train)
validation_set.fit(X_test)


# %% Creacion  del modelo 

modelo = tf.keras.Sequential([
    tf.keras.layers.Conv2D(96,(7,7),padding='same',strides=4,input_shape=(32,32,3),kernel_regularizer='l2'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    
    tf.keras.layers.Conv2D(256,(5,5),padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    
    tf.keras.layers.ZeroPadding2D((1,1)),
    tf.keras.layers.Conv2D(512,(3,3),padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    
    tf.keras.layers.ZeroPadding2D((1,1)),
    tf.keras.layers.Conv2D(1024,(3,3),padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    
    tf.keras.layers.ZeroPadding2D((1,1)),
    tf.keras.layers.Conv2D(1024,(3,3),padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(3072),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.5),
    
    tf.keras.layers.Dense(4096),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.5),
    
    tf.keras.layers.Dense(len(clases)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('softmax')
    
    ])

modelo.summary()
modelo.compile(
    optimizer = tf.keras.optimizers.Adadelta(),
    loss = 'categorical_crossentropy',
    metrics=['accuracy']
    )

# %% Entrenamiento del modelo. Parametros  de AlexNet

history = modelo.fit(
    train_set.flow(X_train,y_train),
    validation_data=validation_set.flow(X_test,y_test),
    epochs=100,steps_per_epoch=len(X_train)//32,validation_steps=len(X_test)//32
    )


#%% Evaluacion del modelo

# Esta es la extructura de AlexNet mejorada 

plt.figure(figsize=(8,8))
plt.subplot(121)
plt.title("Image_1")
plt.imshow(X_train[10].astype('uint8'))
plt.axis('off')
plt.subplot(122)
plt.title("Image_2")
plt.imshow(X_train[100].astype('uint8'))
plt.axis('off')
plt.show()
