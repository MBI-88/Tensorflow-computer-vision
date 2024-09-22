#%% Conceptos tecnicos de CNN

"""
Convolucion:

Es el blocke principal de una CNN. Consiste en la multiplicacion de uan seccion de una 
imagen de entrada por un kernel para producir una salida.

Despues de una convolucion una CNN aprende un poco mas sobre las variables de una imagen
primero se comienza por la deteccion de bordes, depues en la siguiente convolucion aprende 
formas seguido por un mapa de variables.

Durante la operacion de convolucion el tamaño del filtro puede ser cambiado, tipicamente
el numero de filtros es aumentado despues de que la dimesion espacial de un mapa de variables
halla sido decrementada a traves de la convolucion,pooling y stride.

La profundidad del mapa de variables aumenta cunado el tamaño de ventana del filtro aumenta.

Convolucion sobre un volumen con filtro de 3x3:

Una imagen de 7x7x3 se le aplica un kernel de 3x3x3 el resultado es una imagen de 5x5 
con una pofundidad igual al numero de filtros usado (ejemplo 32), se reduce el tamaño de
imagen y se aumenta en profundiad. Cada una de las 27 celdas en el kernel es multiplicada
con sus correspondientes 27 celdas de la imagen de entrada.

Convolucion sobre un volumen con filtro de 1x1:

Un filtro de 1x1 es un fuerte multiplo de una imagen de entrada.
Sobre una imagen de entrada de tamaño 5x5x3 es usado 32 filtro de 1x1x3, la salida de esta 
operacion es un mapa de variables de 5x5x32 , se conserva el ancho y el alto de la imagen de 
entrada solo aumenta la profundidad.
El mismo filtro tambien puede ser usado para decrementar el valor de profundidad de un mapa 
de variables. Ejemplo una mapa de 5x5x128 se le aplica 32 kernels de 1x1 se obtiene a la salida
un mapa de 5x5x32.

Pooling: 

Es la siguiente operacion despues de una convolucion. Es usado para reducir la dimensioalidad y 
el tamaño del mapa de variables sin cambiar la profundidad. El numero de parametros para el pooling
es cero. Los tipos mas populares de pooling son:
-Max Pooling
-Average pooling

En el max poling corremos la ventana sobre el mapa de variables y tomamos el valor maximo de la 
ventana, mientras que en average poling tomamos el valor promedio en la ventana.
Junto a la convolucion eligen la tarea de extraccion de variables.

Padding:

Es usado para preservar el tamaño de el mapa de variables. En la convolucion 2 problemas pueden 
ocurrir y padding resuelve ambos:
-El recorte del mapa de variable con cada convolucion.
-Perdida en la  informacion de los bordes, los pixeles de los  bordes son alterados una vez,
mientras los pixeles del medio son alterados muchas veces por varias operaciones de convolucion.

Stride:

En una convolucion el kernel se mueve una celda (un paso) por cada operacion de convolucion.
El stride permite saltar un paso.
-Cuando el stride = 1, aplicamos una convolucion sin saltos.
-Cuando el stride = 2, saltamos un paso. Esto reduce el tamaño de imagen.


Activation:

La capa de acitvacion adiciona una no linealidad en la red neuronal. Esto es muy importante 
para imagenes y variables dentro de una imagen  es un problema de no linealidad, y muchas otras
funciones dentro de las CNN generan solamente transformaciones lineales. La funcion de activacion 
genera la no linealidad mientras mapea valores del rango de entrada.

Las mas comunes son:
.Sigmoid
.Tanh
.ReLU 

La ReLU tiene algunas ventajas sobre la Tanh y la Sigmoid:
-La funcion sigmoid y tanh tiene el problema del desvanesimineto del gradiente, lento aprendisaje
comparado con relu, como ambas se acercan a uno en un valor de entrada mayor a 3.
-La sigmoid solamente tiene valores positivos para entradas menores que 0.
-La funcion relu es efectiva en la computacion.


Regularizacion:

Es una tecnica para reducir el sobre ajuste.

.L1: Para cada uno de los pesos del modelo w, un parametro adicional lambda|w|, es adicionado 
al objetivo del modelo. Esta regularizacion lleva los factores de pesos a 0 durante la
optimizacion. Penaliza las ponderaciones |w|.

.L2: Para cada uno de los pesos,w,un parametro adicional 1/2*lambda*w^2 es adicionado al objetivo 
del modelo. Esta regularizacion hace que los factores de pesos no sean exactamente 0 durante el proceso de
optimizacion. L2 puede ser esperado para dar un superior rendimiento sobre L1. Penaliza las ponderaciones 
(w^2)

.Max nor constraints: Este tipo de regularizacion adiciona a una bondaridad maxima el  pesos de 
CNN tal que |w| < C, donde C puede ser 3 o 4. Previene a la red del sobre ajuste aun cundo el 
rango de aprendisaje sea alto.

Droput:

Es un tipo especial de regularizacion y se refiere a ignorar  neuronas in una red. Una capa totalmente 
conectada con drop = 0.2, significa que solo el 80% de las neuronas de la capa estan conectadas con la 
siguiente capa. Es una seleccion aleatoria del apagado de las neuronas. Previene a la red de arrancar 
dependiendo de un pequeño numero de neuronas.
.Fuerza que la red aprenda mas variables robustas.
.El tiempo de entrenamiento por epoca es menor, pero el numero de iteraciones se duplica.
.Droput resulta en un aumento de precision de 1-2%.

Bacth normalization:

Direcciona el problema de la conmutacion de covariansa interna por la sustraccion de la media del bloque al 
bloques de la entrada actual y dividiendolo por  un bloque de desviacion estandar. Esta nueva 
entada es entonces multiplicada por el factor de peso actual y adicionado por el termino de bias de la 
salida.
Cuando se aplica un Bacth normalization, se calcula la media de un bloque de informacion de entrada 
y la varianza. Entonces con esta informacion calculamos la normalizacion de la entrada.
La salida es calculada como la escala multiplicada por la entrada normalizada, mas un plus de offset.

Softmax:

Es la funcion de activacion usada  para la capa final de una CNN. Entrega la probailidad de varias clases(+2).
Generalmente es usada cunado se tienen mas de una clase para la clasificacion.

Optimizacion de los Parametros de una CNN:

.Tamaño de imagen
.Filtro. El alto y el ancho.
.Numero de filtros.
.Padding
.Stride 
.Output size
.Numero de parametros.

"""
# %% Librerias
import tensorflow as tf
import zipfile,pathlib,os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import  image
from tensorflow.keras.applications.vgg16 import preprocess_input
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.random.set_seed(42)

# %% Descompresion del archivo
"""
directorio = "C:/Users/MBI/Documents/Python_Scripts/Datasets/Flowers.zip"
path_dir = "C:/Users/MBI/Documents/Python_Scripts/Tensorflow-Vision-Computacional/Cap4-DeepLearning_on_Images"

with zipfile.ZipFile(directorio) as zf:
    zf.extractall(path_dir,pwd=None)
    zf.close()
"""
#%% Procesado de datos

batch_size = 32
path_train = "C:/Users/MBI/Documents/Python_Scripts/Tensorflow_Vision_Computacional/Cap4-DeepLearning_on_Images/flowers_train"
path_validation = "C:/Users/MBI/Documents/Python_Scripts/Tensorflow_Vision_Computacional/Cap4-DeepLearning_on_Images/flowers_test"

train_data = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=90,
    horizontal_flip=True,
    vertical_flip=True
    )

train_generator = train_data.flow_from_directory(path_train, target_size=(64,64),batch_size=batch_size)

validation_data = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=90,
    horizontal_flip=True,
    vertical_flip=True
    )
validation_generator = validation_data.flow_from_directory(path_validation,target_size=(64,64),batch_size=batch_size)

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

modelo = tf.keras.Sequential([
    tf.keras.layers.Conv2D(96,11,padding='valid',activation='relu',input_shape=(64,64,3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(256,5,padding='same',activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(384,3,padding='same',activation='relu'),
    tf.keras.layers.Conv2D(384,3,padding='same',activation='relu'),
    tf.keras.layers.Conv2D(256,3,padding='same',activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(1024,3,padding='same',activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4096,activation='relu'),
    tf.keras.layers.Dense(5)
     ])

adam = tf.keras.optimizers.Adam(lr = 0.0001)
modelo.compile(
    optimizer=adam,
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
    
    )
modelo.summary()
#%% Entrenamiento del modelo

historial = modelo.fit(
    train_generator,
    epochs=25,
    steps_per_epoch=num_total_train//batch_size,
    validation_data=validation_generator,
    validation_steps=num_total_test//batch_size)

#%% Prueba del modelo

image_path = "C:/Users/MBI/Documents/Python_Scripts/Tensorflow_Vision_Computacional/Cap4-DeepLearning_on_Images/flowers_test/tulip/11746080_963537acdc.jpg"
img = image.load_img(image_path,target_size=(64,64))
img = image.img_to_array(img)
img = tf.expand_dims(img,axis=0)
img = preprocess_input(img)

pred = modelo.predict(img)
plt.figure(figsize=(10,8))
plt.subplot(1,2,1)
plt.title("Prediccion",color='red')
plt.imshow(pred)
plt.axis('off')
plt.subplot(122)
plt.title("Imagen",color='blue')
plt.imshow(img[0])
plt.axis('off')
plt.show()
print(img.shape)



#%% Visualizando las capas

# Se crea un nuevo modelo a partir del ya creado
layer_outputs = [layer.output for layer in modelo.layers[:len(modelo.layers)]]
activtion_modelfig = tf.keras.Model(inputs=modelo.input,outputs=layer_outputs)
activation_layers = activtion_modelfig.predict(img)

for i in range(0,len(modelo.layers)-3):
    curren_layer = activation_layers[i]# Accediendo a la informacion de cada capa
    ns = curren_layer.shape[-1] # canales del mapa de variables (kernel)
    figure = plt.figure(figsize=(12,8))
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

# Nota: En el lazo se muestra para cada capa el filtro en inicio,medio y final.Se observa en cada imagen la extracion de variables hechas por los kernels









