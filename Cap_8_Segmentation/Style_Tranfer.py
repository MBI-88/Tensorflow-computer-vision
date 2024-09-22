"""
Created on Fri Jul 15 15:32:55 2021
@author:MBI

Entendiendo las transferencia de estilo

La transferencia de estilo es una tecnica donde se mescla el contenido de una imagen y el estilizado de otra por
las iguladad de las distribucion de sus variables para generar la imagen final que es similar al contenido de la
imagen de referencia pero con una estilo artisticamente diferente.

Pasos para su relaizacion:
- Seleccionar el modelo VGG19 que tiene 5 redes de convolucion con 4 capas por convolucion, seguido de una capa
completamente conectada.
- Cargar el contenido de una imagen a traves de la red VGG19.
-Predecir las 5 ultimas capas convolucionales.
-Cargar la VGG19 sin las ultimas capas, listar sus nombres.
- La capa convolucional en VGG19 hace la extracion de variables, mientras las completamente conectadas se encargan de
las tareas de clasificacion. Sin la ultima capa, la red solamente tendra las 5 ultimas capas convolucionales. Las capas
iniciales transmiten los pixeles de entrada de las filas de las imagen, mientras las capas finales capturan la definicion
de variables ya partes de la imagen.
- Haciendo esto, el contenido de una imagen es representado por el mapa de variables intermedio, en este caso en particular
seria el 5 bloque de convolucion.
-El promedio maximo despreciable (MMD) es usado para comparar 20 vectores. El estilo de una imagen es representado por la
matrix gram  de el mapa de variables del estilo de una imagen. La matrix gram nos da las relacion entre vectores de variables
y es representado por el producto dot. Esto pude tambien estar a traves de la correlacion entre el vector de variables y el
promedio de valores sobre la imagen de entrada.
-Las perdidas totales = perdidas de estilo + perdidas de contenido. La perdida es calculada como la suma de los pesos del error
cuadratico medio de la imagen de salida relativa a la imagen objetivo.

Matrix Gram: Es una matrix de los productos de entrada G= Ii*Ij^T. Donde Ii y Ij son los vectores de variables de la imagen original
y las imagen estilizada. El producto representa la covarianza de los vectores, los cuales representan correlacion.

"""
# %% Modulos
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.models import Model
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import PIL


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# %% Preparando los datos
style_image_path = "Tensorflow_Vision_Computacional/Cap_8_Segmentation/campo-trigo-van-gogh.jpg"
conten_image_path = "Tensorflow_Vision_Computacional/Cap_8_Segmentation/dog-01.jpg"


def preprocess_img(ruta=''):
    if ruta == '':
        return print("[-]No hay ruta de imagen")
    else:
        max_dim = 2048
        image = tf.io.read_file(ruta)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        shape = tf.cast(tf.shape(image)[:-1], dtype=tf.float32)
        long_dim = max(shape)
        scale_dim = long_dim / max_dim
        new_shape = tf.cast(shape * scale_dim, tf.int32)
        image = tf.image.resize(image, new_shape)
        image = image[tf.newaxis, :]
        return image


def show_image(image, title=''):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)
    plt.title(title)
    plt.imshow(image)
    plt.axis('off')


# %% Muestra de imagenes

style_image = preprocess_img(style_image_path)
content_image = preprocess_img(conten_image_path)

plt.figure(figsize=(12, 8))
plt.subplot(121)
show_image(content_image, "Imagenes de referencia")
plt.subplot(122)
show_image(style_image, "Imagen de estilo")
plt.show()
# %% Cargar el modelo VGG19

model_vgg19 = VGG19(include_top=False, weights="imagenet")
for layer in model_vgg19.layers:
    print(layer.name)

# %% Contruyendo los extractores de estilo

style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
content_layers = ['block5_conv2']


def vgg19_extractor(layers_name):
    model_vgg19.trainable = False
    outputs = [model_vgg19.get_layer(name).output for name in layers_name]
    modelo_final = Model(inputs=[model_vgg19.inputs], outputs=outputs)
    return modelo_final


style_extractor = vgg19_extractor(style_layers)
style_output = style_extractor(style_image * 255)

for name, output in zip(style_layers, style_output):
    print(name, "\n")
    print("Sahpe: ", output.numpy().shape)
    print("Minimum: ", output.numpy().min())
    print("Maximum: ", output.numpy().max())
    print("Mean: ", output.numpy().mean())


# %%  Construyendo la calse Styler

class Styler(Model):
    def __init__(self, style_layers, content_layers):
        super(Styler, self).__init__()
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_layers = len(style_layers)
        self.num_conten = len(content_layers)
        self.vgg19 = vgg19_extractor(style_layers + content_layers)
        self.vgg19.trainable = False

    def call(self, inputs):
        inputs = inputs * 255.0
        preprocess_inputs = preprocess_input(inputs)
        outputs = self.vgg19(preprocess_inputs)
        style_outputs, content_outputs = (outputs[:self.num_layers], outputs[self.num_layers:])
        style_outputs = [self.gram_matrix(style) for style in style_outputs]
        content_dic = {content_name: valor for content_name, valor in zip(self.content_layers, content_outputs)}
        style_dic = {style_name: valor for style_name, valor in zip(self.style_layers, style_outputs)}
        return {'content': content_dic, 'style': style_dic}

    def gram_matrix(self, input_tensor):
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_local = tf.cast(input_shape[1] * input_shape[2], tf.float32)
        return result / num_local


# %% Evaluando al extractor

extractor = Styler(style_layers, content_layers)
result = extractor(tf.constant(content_image))
print("Style")
for name, output in sorted(result['style'].items()):
    print(name, "\n")
    print("Sahpe: ", output.numpy().shape)
    print("Minimum: ", output.numpy().min())
    print("Maximum: ", output.numpy().max())
    print("Mean: ", output.numpy().mean())

print("Conten")
for name, output in sorted(result['content'].items()):
    print(name, "\n")
    print("Sahpe: ", output.numpy().shape)
    print("Minimum: ", output.numpy().min())
    print("Maximum: ", output.numpy().max())
    print("Mean: ", output.numpy().mean())
# %% Optimizador del modelo

style_target = extractor(style_image)['style']
content_target = extractor(content_image)['content']
image = tf.Variable(content_image)


def clip_image(imge):
    return tf.clip_by_value(imge, 0.0, 1.0)


optimizer = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
style_weights = 1e-2
content_weights = 1e4


def style_conten_loss(outputs):
    style_output = outputs['style']
    content_output = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_output[name] - style_target[name]) ** 2)
                           for name in style_output.keys()])
    content_loss = tf.add_n([tf.reduce_mean((content_output[name] - content_target[name]) ** 2)
                             for name in content_output.keys()])

    style_loss *= style_weights / len(style_layers)
    content_loss *= content_weights / len(content_layers)
    losses = style_loss + content_loss
    return losses


# %% Entrenamiento

def tensor_to_image(tensor):
    tensor = tensor * 255
    image = np.array(tensor, dtype='uint8')
    if np.ndim(image) > 3:
        assert image.shape[0] == 1
        image = image[0]
    return PIL.Image.fromarray(image)


lista_image = []


@tf.function
def gradiente_step(image):
    with tf.GradientTape() as tape:
        output = extractor(image)
        loss = style_conten_loss(output)

    grad = tape.gradient(loss, image)
    optimizer.apply_gradients([(grad, image)])
    image.assign(clip_image(image))


epoch = 10
step_per_epoch = 100
step = 0
for n in range(epoch):
    for m in range(step_per_epoch):
        step += 1
        gradiente_step(image)
    lista_image.append((tensor_to_image(image), "Train step step: {}".format(step)))

plt.figure(figsize=(19, 15))
for i, img in enumerate(lista_image, 1):
    plt.subplot(5, 5, i)
    plt.title(img[1])
    plt.imshow(img[0])
    plt.axis('off')
plt.show()


# %% Variacion total de la perdida ejemplo

def high_pass_variation(image):  # Esta funcion cumple la misma funcion de un filtro sobel
    x_var = image[:, :, 1:, :] - image[:, :, :-1, :]  # shape(batch,y,x,cha)
    y_var = image[:, 1:, :, :] - image[:, :-1, :, :]
    return x_var, y_var


x_delta, y_delta = high_pass_variation(content_image)

plt.figure(figsize=(14, 10))
plt.subplot(221)
plt.title("variacion y")
show_image(2 * y_delta + 0.5, "Variacion horizontal")
plt.subplot(222)
show_image(2 * x_delta + 0.5, "Variacion vertical")
x_delta, y_delta = high_pass_variation(image)
plt.subplot(223)
plt.title("variacion y")
show_image(2 * y_delta + 0.5, "Variacion horizontal estilizada")
plt.subplot(224)
show_image(2 * x_delta + 0.5, "Variacion vertical estilizada")
plt.show()
# %% Usando la variacion total en el entrenamiento

img = tf.Variable(content_image)
total_variation_weights = 30

lista_image = []


@tf.function
def gradiente_step(image):
    with tf.GradientTape() as tape:
        output = extractor(image)
        loss = style_conten_loss(output)
        loss += total_variation_weights * tf.image.total_variation(image)

    grad = tape.gradient(loss, image)
    optimizer.apply_gradients([(grad, image)])
    image.assign(clip_image(image))


epoch = 10
step_per_epoch = 100
step = 0
for n in range(epoch):
    for m in range(step_per_epoch):
        step += 1
        gradiente_step(image)
    lista_image.append((tensor_to_image(image), "Train step step: {}".format(step)))

plt.figure(figsize=(19, 15))
for i, img in enumerate(lista_image, 1):
    plt.subplot(5, 5, i)
    plt.title(img[1])
    plt.imshow(img[0])
    plt.axis('off')
plt.show()

tensor_to_image(image).save("Tensorflow_Vision_Computacional/Cap_8_Segmentation/Style_Image_2.png")
