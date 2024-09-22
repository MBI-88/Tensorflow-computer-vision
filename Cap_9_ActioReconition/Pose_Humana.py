"""
Create Jul on 22 17:55:10
@author: MBI

Pasos importantes para el algoritmo de Pose Humana:
. La entrada para la red consiste de las 10 primeras capas del modelo VGG19, el cual es usado para gene-
rar un set de mapas de variables.
. La red OpenPose toma los mapas de variables como entrada y consiste de 2 estados de CNN:-Par Affinity
Fields (PAF) con Tp numero de iteraciones. El modelo OpenPose presentado en el 2018 es un mejoramiento en
general de todo el modelo anterior presentado  en el 2017 devido a 2 metricas claves:
-Reduce el timepo de computacion por la mitad calculando el PAF y un mapa de confidencialidad. Estas
diferencias de el calculo simultaneo de ambos y remplazando la convolucion 7X7 con una de 3X3
-Mejora la precision mejorando el mapa de confidencialidad sobre el regualr PAF (2017) aumentando la pro-
fundiad de la red.

. En el siguiente estado, la prediccion de los estados previos y las variables de la imagne original,F,son
concatenadas para producir puntos de 2 dimensiones para toda las personas en la imagem. La funcion de perdida
es aplicada en el final de cada estado  entre la prediccion estimadoa , la objetiva y el PAF. Este proceso
es representado pro muchas iteraciones , resultando en el mapa de variable mas actulizado de la deteccion
PAF.

El modelo openPose de OpneCv se encuntra en https://github.com/quanhua92/human-pose-estimation-opencv.
"""
#%% Modulos
import cv2 as cv
import argparse as arg
#%% Algoritmo

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]


parcer = arg.ArgumentParser()
parcer.add_argument("--input",help='Direccion par imagen o video. Empezar capturar de una camara')
parcer.add_argument('--threshold',default=0.2,type=float,help='Valor para separar la figura de la imagen')
parcer.add_argument('--width',default=368,type=int,help='Ancho de la imagen de entrada')
parcer.add_argument('--height',default=368,type=int,help='Alto de la imagne de entrada')

args = parcer.parse_args()

inWidth = args.width
inHeight = args.height
inThreshold = args.threshold

net = cv.dnn.readNetFromTensorflow('Tensorflow_Vision_Computacional/Cap_9_ActioReconition/graph_opt.pb')

cap = cv.VideoCapture(args.input if args.input else 0)

while cv.waitKey(1) < 0:
    hasfram,frame = cap.read()
    if  not hasfram:
        cv.waitKey()
        break
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    net.setInput(cv.dnn.blobFromImage(frame,1.0,(inWidth,inHeight),(127.5,127.5,127.5),swapRB=True,crop=False))
    out = net.forward()
    out = out[:,:19,:,:] # MobileNet output [1, 57, -1, -1], solo necesitamos 19 elementos
    assert(len(BODY_PARTS) == out.shape[1])

    points = []

    for i in range(len(BODY_PARTS)):
        heatmap = out[0,i,:,:]
        _,conf,_,point = cv.minMaxLoc(heatmap)
        X = (frame_width * point[0]) / out.shape[3]
        Y = (frame_height * point[1]) / out.shape[2]
        points.append((int(X),int(Y)) if conf > inThreshold else None)

    for pose in POSE_PAIRS:
        partFrom  = pose[0]
        partTo = pose[1]
        assert (partFrom in BODY_PARTS)
        assert (partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

    t,_ = net.getPerfProfile()
    freq = cv.getTickFrequency() / 1000
    cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    cv.imshow('OpenPose using OpenCV', frame)

