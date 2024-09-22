#%% Librerias

from matplotlib import colors
import numpy as np 
import  cv2
import matplotlib.pyplot as plt 
from PIL import Image
from skimage.util import random_noise
from skimage.feature import hog
from skimage import data,exposure
# %% Detectando bordes usando hashing
"""
El metodo hashing es usado para encontrar similitudes entre imagenes. Modifia una imagen
de entrada a un vector binario ajustado. Despues de la transformacion la imagen pude ser
comparada rapidamente con la distancia de Hamming. Una distacia de 0 indica total similitud
mientras que una distacia diferente de 0 indica imagenes diferentes.
"""

# %% Creando un vector de imagen

image = Image.open("Cap_1-ImagenProcesing\\car2.png")
image_arra = np.asarray(image)
print(image_arra.shape)

gray = cv2.cvtColor(image_arra,cv2.COLOR_BGR2GRAY)
plt.figure(figsize=(15,5))
plt.subplot(231)
plt.title("Original image")
plt.imshow(image)
plt.axis('off')
plt.subplot(232)
plt.title("Gray image")
plt.imshow(gray,cmap="gray")
plt.axis('off')
plt.subplot(234)
plt.title("Red channel")
plt.imshow(image_arra[:,:,0]) # color rojo solamente
plt.axis('off')
plt.subplot(235)
plt.title("Grenn")
plt.imshow(image_arra[:,:,1]) # color verde solamente
plt.axis('off')
plt.subplot(236)
plt.title("Blue")
plt.imshow(image_arra[:,:,2]) # color azul solamente
plt.axis('off')
plt.show()
# %% Filtros lineales, convolucion con kernels (ventanas)

img = Image.open("Cap_1-ImagenProcesing\\car3.png")
array = np.asarray(img)
print(array.shape)
gray = cv2.cvtColor(array,cv2.COLOR_BGR2GRAY)
kernels = np.array([[-1,-1,-1],[2,2,2],[-1,-1,-1]]) # kernel horizontal

blurimg = cv2.filter2D(gray,-1,kernels)

plt.figure(figsize=(10,5))
plt.subplot(121)
plt.title("Original")
plt.imshow(img)
plt.axis('off')
plt.subplot(122)
plt.title("Img blurimg")
plt.imshow(blurimg,cmap="gray")
plt.axis('off')
plt.show()

# %% Efecto del filtrado de imagen

img = cv2.imread("Cap_1-ImagenProcesing\\car3.png")
imgnoise = random_noise(img,mode="s&p",amount=0.3)
plt.imsave("Cap_1-ImagenProcesing\\noisecar.jpg",imgnoise)
imgnew = cv2.imread("Cap_1-ImagenProcesing\\noisecar.jpg")
filtermean = cv2.blur(imgnew,(3,3))
filtermedian = cv2.medianBlur(imgnew,3)
filtergaussian = cv2.GaussianBlur(imgnew,(3,3),0)

plt.figure(figsize=(10,5))
plt.subplot(221)
plt.title("Original image noise")
plt.imshow(imgnew)
plt.axis('off')
plt.subplot(222)
plt.title("Mean filter")
plt.imshow(filtermean)
plt.axis('off')
plt.subplot(223)
plt.title("Median filter")
plt.imshow(filtermedian)
plt.axis('off')
plt.subplot(224)
plt.title("Gaussian filter")
plt.imshow(filtergaussian)
plt.axis('off')
plt.show()

# %% Gradiente de imagen
"""
 Ejemplo de kernel horizontal: [[-1,-1,-1],
                                [2,2,2],
                                [1,-1,-1]]

Ejemplo de kernel vertical: [[-1,2,-1],
                             [-1,2,-1],
                             [-1,2,-1]]

Ejemplo de kernel oblicuo de 45 grados : [[-1,-1,2], 
                                          [-1,2,-1],
                                          [2,-1,-1] ]

El gradiente de imagen es el concepto fundamental de la vision computacional
.el gradiente puede ser calculado tanto en x-axis como y-axis
.usando el gradiente de imagen se determinan bordes y esquinas
.los paquetes de bordes y esquinas tienen mucha informacion  sobre la forma o variabel 
de una imagen
.entonces el gradiente de una imagen es un mecanismo que convierte un pixel de bajo orden
de informacion a uno de alto orden de informaicon de la variabel de imagen 

Nitides de una imagen

.La baja frecunecia de ruido de una imagen es removida por la aplicacion de un filtro 
paso alto (operador diferencia) lo cual resulta en una estructura de linea y bordes que
se vuelven mas visibles.Esto tambien es conocido como la operacion Laplace, la cual usa la
segunda derivada con respecto a los ejes coordenados.

Las cuatro celdas adjacentes al punto medio del kernel siempre tienen signos opuestos
ejemplo de kernel:
  [[0,-1,0],    
   [-1,4,-1],
   [0,-1,0]]    

  [[0,1,0],
   [1,-4,1],
   [0,1,0]]


Como se usan las operaciones

Una imagen consiste de variables, caracteristicas y objetos que no son variables de la 
imagen.Reconocimineto de imagenes es todo sobre la extraccion de variables descriptoras
de una imagen dada y la eliminacion de lo que no sea variables de la imagen.Un filtro 
gaussiano es el metodo de suprimir las  no variables de una variables en una imagen (ruido)
Aplicando el filtro gaussino varias veces se condigue la supresion de estos elementos no de-
seados.Despues de esto las variables son fuertes y pueden ser extraidas por el metodo del gra-
diente de laplace.Por esta rason convolucionamos multiples veces para distinguir las variables. 

Sobel y Canny son los metodos de deteccion de bordes de 1er orden y Laplace es el metodo de dete-
ccion de 2do orden.
"""

# %% Detector Sobel y Canny

img = cv2.imread("Cap_1-ImagenProcesing\\car3.png",cv2.IMREAD_GRAYSCALE)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
mag,direction = cv2.cartToPolar(sobelx,sobely,angleInDegrees=True)
sobelG = np.hypot(sobelx,sobely) # Calculo de la hipotenusa del triangulo

canny = cv2.Canny(img,100,200)

plt.figure(figsize=(15,5))
plt.subplot(231)
plt.title("Sobel-x")
plt.imshow(sobelx,cmap='gray')
plt.axis('off')
plt.subplot(232)
plt.title("Sobel-y")
plt.imshow(sobely,cmap='gray')
plt.axis('off')
plt.subplot(233)
plt.title("Canny")
plt.imshow(canny,cmap='gray')
plt.axis('off')
plt.subplot(234)
plt.title("Sobel G")
plt.imshow(sobelG,cmap='gray')
plt.axis('off')
plt.subplot(235)
plt.title("Mag")
plt.imshow(mag,cmap='gray')
plt.axis('off')
plt.subplot(236)
plt.title("Direction (angle)")
plt.imshow(direction,cmap='gray')
plt.axis('off')
plt.show()

# %% Extraccion de variables de una imagen

image = Image.open("car2.png")
image_arra = np.asanyarray(image)
image_arra.shape
colors = ('blue','green','red')

plt.figure(figsize=(10,5))
plt.subplot(121)
plt.title("Car")
plt.imshow(image)
plt.axis('off')
plt.subplot(122)
plt.title("Histogram")
for i,his in enumerate(colors):
  carhistogram = cv2.calcHist([image_arra],[i],None,[256],[0,256])
  plt.plot(carhistogram,color=his)
  plt.xlim([0,256])
plt.xlabel("Bright intecity")
plt.ylabel("Frecuency")
plt.show()

# %% Igualando imagenes usando OpenCv 

tile = cv2.imread("tile.jpeg",0)
bath = cv2.imread("bathroom_image.jpeg",0)

orb = cv2.ORB_create()
k1,des1 = orb.detectAndCompute(tile,None)
k2,des2 = orb.detectAndCompute(bath,None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)

matches = bf.match(des1,des2)
matches = sorted(matches,key=lambda x:x.distance)

draw_params = dict(matchColor=(0,255,0),singlePointColor=(0,255,0))

floor_matches = cv2.drawMatches(tile,k1,bath,k2,matches[:50],None,flags=2,**draw_params)

sift = cv2.SIFT_create()
k1,des1 = sift.detectAndCompute(tile,None)
k2,des2 = sift.detectAndCompute(bath,None)

bf = cv2.BFMatcher()
match_falnn = bf.knnMatch(des1,des2,k=2)
good = []
for mat1,mat2 in match_falnn:
  if  mat1.distance < 1.2*mat2.distance:
    good.append([mat1])

draw_params = dict(matchColor=(0,0,255),singlePointColor=(0,0,255))

sift_matches = cv2.drawMatchesKnn(tile,k1,bath,k2,good,None,**draw_params)

sift = cv2.SIFT_create()
k1,des1 = sift.detectAndCompute(tile,None)
k2,des2 = sift.detectAndCompute(bath,None)

flann_index = 0
index_params = dict(algorithm=flann_index,trees=7)
search_para = dict(check=50)

flann = cv2.FlannBasedMatcher(index_params,search_para)
matches = flann.knnMatch(des1,des2,k=2)

matches_mask = [[0,0] for i in range(len(matches))]

for i,(match1,match2) in enumerate(matches):
    if match1.distance <1.2*match2.distance:
        matches_mask[i]=[1,0]

draw_params = dict(matchColor=(255,0,0),singlePointColor=(0,0,255),matchesMask=matches_mask,flags=2)

flann_matches = cv2.drawMatchesKnn(tile,k1,bath,k2,matches,None,**draw_params)

plt.figure(figsize=(12,5),facecolor="gray")
plt.subplot(131)
plt.title("ORB  matches")
plt.imshow(floor_matches,cmap="gray")
plt.axis('off')
plt.subplot(132)
plt.title("Sift matches")
plt.imshow(sift_matches,cmap="gray")
plt.axis('off')
plt.subplot(133)
plt.title("Flann matches")
plt.imshow(flann_matches,cmap='gray')
plt.axis('off')
plt.show()

# %% Deteccion de objetos usando contornos y el detector HOG

img = cv2.imread("appleorange.jpg")
img_copy = img.copy()
thresh = 100
canny_output = cv2.Canny(img,threshold1=thresh,threshold2=thresh*2)
contours,hierarchy = cv2.findContours(canny_output,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

count = 0
for c in contours:
  x,y,w,h = cv2.boundingRect(c)
  if (w > 20 and h > 20):
    count += 1
    roi = img[y+int(h/4):y+int(3*h/4),x+int(h/4):x+int(3*h/4)]
    roi_meancolor = cv2.mean(roi)
    if (roi_meancolor[0] > 30 and roi_meancolor[0] < 40 and roi_meancolor[1] > 70 and roi_meancolor[1] < 105 and roi_meancolor[2] > 150 and roi_meancolor[2] < 200):
      cv2.putText(img,"orange",(x-2,y-2),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2,cv2.LINE_AA)
      cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)
    else:
      cv2.putText(img,"apple",(x-2,y-2),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2,cv2.LINE_AA)
      cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)


plt.figure(figsize=(12,5),facecolor="white")
plt.subplot(121)
plt.title("AppleOrange")
plt.imshow(img_copy[:,:,::-1])
plt.axis('off')
plt.subplot(122)
plt.title("Contours detected")
plt.imshow(img[:,:,::-1])
plt.axis('off')
plt.show()


# %% Detector Hog

img = cv2.imread("appleorange.jpg")
fruit,hog_img = hog(img,orientations=8,pixels_per_cell=(16,16),cells_per_block=(1,1),visualize=True,multichannel=True)
hog_image_rescaled = exposure.rescale_intensity(hog_img,in_range=(0,10))

plt.figure(figsize=(14,5),facecolor='gray')
plt.subplot(121)
plt.title("AppleOrange")
plt.imshow(img[:,:,::-1])
plt.axis('off')
plt.subplot(122)
plt.title("Detection")
plt.imshow(hog_image_rescaled,cmap="gray")
plt.axis('off')
plt.show()



# %% Variando los objetos de la imagen

imagen = cv2.imread("appleorangeother.jpg")
imagencopy = imagen.copy()

output_canny = cv2.Canny(imagen,100,200)
contours,hierarchy = cv2.findContours(output_canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
  x,y,w,h = cv2.boundingRect(c)
  if (w > 60 and w < 100 and h > 60 and h < 120):
    roi = imagen[y+h//4:y+(3*h//4),x+h//4:x+(3*h//4)]
    roi_m = cv2.mean(roi)
    if (roi_m[0] > 10 and roi_m[0] < 40 and roi_m[1] > 65 and roi_m[1] < 105):
      cv2.putText(imagen,"Orange",(x-2,y-2),cv2.FONT_HERSHEY_PLAIN,0.88,(36,250,20),1,cv2.LINE_AA)
      cv2.rectangle(imagen,(x,y),(x+w,y+h),(255,0,0),1,cv2.LINE_AA)
    else:
      cv2.putText(imagen,"Apple",(x-2,y-2),cv2.FONT_HERSHEY_PLAIN,0.88,(0,255,250),1,cv2.LINE_AA)
      cv2.rectangle(imagen,(x,y),(x+w,y+h),(20,230,100),1,cv2.LINE_AA)

plt.figure(figsize=(15,8),facecolor='gold')
plt.subplot(121)
plt.title("Original")
plt.imshow(imagencopy[:,:,::-1])
plt.axis('off')
plt.subplot(122)
plt.imshow(imagen[:,:,::-1])
plt.axis('off')
plt.show()

# %% Usando Hog en la nueva imagen

frutas,hog_image = hog(imagencopy,8,(16,16),(1,1),visualize=True,multichannel=True)
hog_image_rescaled = exposure.rescale_intensity(hog_image,in_range=(0,15))

plt.figure(figsize=(15,8),facecolor='white')
plt.subplot(121)
plt.title("AppleOrange")
plt.imshow(imagen[:,:,::-1])
plt.axis('off')
plt.subplot(122)
plt.title("Detection")
plt.imshow(hog_image_rescaled,cmap="gray")
plt.axis('off')
plt.show()

