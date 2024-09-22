# %% Librerias
from scipy.ndimage.measurements import histogram
from skimage.feature import local_binary_pattern
from skimage.transform import rotate
from skimage import data
from skimage.color import label2rgb
import matplotlib.pyplot as plt
import numpy as np 
import cv2 ,os
from scipy.stats import wasserstein_distance
from scipy import spatial,ndimage
from PIL import Image
# %% Aplicando LBP al reconocimiento de texturas (Metodo para reconocimento de texturas.Machine-Learning)

path_train = "patterndata/train"
path_test = "patterndata/test"
radius = 5
num_points = 25
match = 0

lbp_train_feature = []
train_data = []
for filename in os.listdir(path=path_train):
    img_train = os.path.join(path_train,filename)
    print(img_train)
    trainingimg = Image.open(img_train)
    trainingiarr = np.asanyarray(trainingimg)
    traininggray = cv2.cvtColor(trainingiarr,cv2.COLOR_BGR2GRAY)
    train_lbp = local_binary_pattern(traininggray,num_points,radius,method="uniform")
    lbp_train_feature.append(train_lbp)
    train_data.append(filename)
# %%  Cargando las muestras de prueba

for filename in os.listdir(path_test):
    img_test = os.path.join(path_test,filename)
    print(img_test)
    testimg = Image.open(img_test)
    test_arr = np.asanyarray(testimg)
    test_gray = cv2.cvtColor(test_arr,cv2.COLOR_BGR2GRAY)
    test_lbp_features = local_binary_pattern(test_gray,num_points,radius,method="uniform")
    bin_num = int(test_lbp_features.max() + 1)
    test_hist,_ = np.histogram(test_lbp_features,bins=bin_num,range=(0,bin_num),density=None)
    mymet = 0
    match = 0
    q = []
    min_score = 10000 # valor de merito

    fig = plt.figure(figsize=(12,5),facecolor="gold")
    fig.subplots_adjust(wspace=.5)
    ax1 = fig.add_subplot(231)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    plt.xticks([])
    plt.yticks([])
    ax1.set_title(filename,fontsize=15, fontweight = 'bold')
    plt.subplot(231), plt.imshow(testimg)
    fig.subplots_adjust(hspace=.5)
    plt.subplot(232),plt.plot(test_hist) 
    ax3 = fig.add_subplot(233)
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])
    plt.xticks([])
    plt.yticks([])
    plt.subplot(233), plt.imshow(test_lbp_features, cmap ='gray')

    ax2 = fig.add_subplot(234)
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    plt.xticks([])
    plt.yticks([])
    
    ax4 = fig.add_subplot(236)
    ax4.set_xticklabels([])
    ax4.set_yticklabels([])
    plt.xticks([])
    plt.yticks([])
    
    # Evaluacion (comparacion de histogramas)

    for features in lbp_train_feature:
        mymet += 1
        histogram,_ = np.histogram(features,bins=bin_num,range=(0,bin_num),density=None)
        p = np.asanyarray(test_hist)
        q = np.asanyarray(histogram)
        filter_indx = np.where(np.logical_and(p != 0,q != 0)) 

        minima = np.minimum(test_hist,histogram)
        interseption = np.true_divide(np.sum(minima),np.sum(histogram))
        #difidx = p - q
        score = .5*(np.sum((p-q)**2/(p+q+1e-10)))
        scorewas = wasserstein_distance(p,q)
        scorechi = .5*(np.sum((p-q)**2/(p+q+1e-10)))
        scoresimple = np.sum((p-q)**2)
        scoreseucl = spatial.distance.euclidean(p,q)
        scoresman = spatial.distance.cityblock(p,q)
        if score < min_score:
            min_score = score
            match = mymet - 1
    
    dir_train = os.path.join(path_train,train_data[match])
    matchedimg = Image.open(dir_train)
    matchedimg_aar = np.asanyarray(matchedimg)
    matchedimg_gray = cv2.cvtColor(matchedimg_aar,cv2.COLOR_BGR2GRAY)
    match_train_lbp = local_binary_pattern(matchedimg_gray,num_points,radius,method="uniform")

    print("\033[1m" + "Match pattern: ", train_data[match], filename + "\033[0m")
    plt.subplot(234),plt.imshow(matchedimg)
    ax2.set_title(train_data[match],fontsize=15, fontweight = 'bold')
    plt.subplot(236),plt.imshow(match_train_lbp, cmap ='gray')
    plt.subplot(235),plt.plot(histogram) 
    plt.show()
# %% Aplicando LBP para igualar color de rostros 

path_face = "facedata/facecolr"
path_foundation = "facedata/foundcolr"

fundation  = []
fundation_data = []

for filename_f in os.listdir(path_foundation):
    img_path = os.path.join(path_foundation,filename_f)
    print(img_path) 
    img_fundation = Image.open(img_path)
    img_fu_asarray = np.asanyarray(img_fundation)
    fundation.append(img_fu_asarray)
    fundation_data.append(filename_f)

for filename_face in os.listdir(path_face):
    img_face = os.path.join(path_face,filename_face)
    img_face_ = Image.open(img_face)
    img_face_assarray = np.asanyarray(img_face_)
    img_face_gray = cv2.cvtColor(img_face_assarray,cv2.COLOR_BGR2GRAY)
    mfs,sfc = cv2.meanStdDev(img_face_assarray)
    static = np.concatenate([mfs,sfc]).flatten()
    face_bchannel = static[0]
    face_gchannel = static[1]
    face_rchannel = static[2]
    minsimilarity = 10000 # valor de merito
    simidx =  0
    matchidx = 0
    figshade = plt.figure()
    figshade.subplots_adjust(wspace=.5)
    axshd1 = figshade.add_subplot(121)
    axshd1.set_xticklabels([])
    axshd1.set_yticklabels([])
    plt.xticks([])
    plt.yticks([])
    axshd1.set_title(filename_face,fontsize=15, fontweight = 'bold')
    plt.subplot(121), plt.imshow(img_face_) 

    axshd2 = figshade.add_subplot(122)
    axshd2.set_xticklabels([])
    axshd2.set_yticklabels([])

    plt.xticks([])
    plt.yticks([])

    for shades in fundation:
        simidx += 1
        mfnd,sfnd = cv2.meanStdDev(shades)
        stat = np.concatenate([mfnd,sfnd]).flatten()
        fundation_bchannel = stat[0]
        fundation_gchannel = stat[1]
        fundation_rchannel = stat[2]
        dif_face_fnd = abs(0.299*(face_rchannel - fundation_rchannel) + 0.587*(face_gchannel - fundation_gchannel)+0.114*(face_bchannel - fundation_bchannel))

        if dif_face_fnd < minsimilarity:
            minsimilarity = dif_face_fnd
            matchidx = simidx - 1
    
    dir_foundation = os.path.join(path_foundation, fundation_data[matchidx])
    matchedfnd = Image.open(dir_foundation)
    
    print("\033[1m" + "Match pattern: ", fundation[matchidx], filename_face + "\033[0m")

    axshd2.set_title(fundation[matchidx],fontsize=15, fontweight = 'bold')
    plt.subplot(122),plt.imshow(matchedfnd)
    
    plt.show()
# Nota: este metodo no es buneo para el reconosimiento de colores de rostros
# %%
