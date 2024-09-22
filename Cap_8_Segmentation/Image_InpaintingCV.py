"""
Created on Ther Jul 15 11:32:55 2021
@author:MBI

Description de uso en opencv de los metodos de inpainting para imagenes:
cv2.INPAINT_TELEA : Este metodo remplaza los pixeles en el vecindario para ser rellenados por la
suma de  pesos normalizados del todos los pixeles conocidos en le vecindario. Mas pesos son dados a esos pixeles que
se encuentran cerca del punto ya del las bondaridades del los contornos.

cv2.INPAINT_NS : Este metodo junta puntos con la misma intencidad mientras iguala vectores gradientes en las bondaridad
de la region del relleno.

"""

#%% Modulos
import  cv2
#%% Main()
# Metodo (Telea) y (NS)

imge = cv2.imread("Tensorflow_Vision_Computacional/Cap_8_Segmentation/WIN_20200713_23_30_26_Pro (2).jpg")
mask = cv2.imread("Tensorflow_Vision_Computacional/Cap_8_Segmentation/Mask.jpg",0)
dst_telea = cv2.inpaint(imge,mask,3,cv2.INPAINT_TELEA)
dst_ns = cv2.inpaint(imge,mask,3,cv2.INPAINT_NS)
cv2.imshow("Image input",imge)
cv2.imshow("Output Telea",dst_telea)
cv2.imshow("Output ns",dst_ns)


while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
# Nota: A partir de una mascara con la misma dimension de la imagen de entrada y teniendo el mismo patron de dibujo
# el metodo pude predecir los pixeles faltantes en la imagen de entrada



