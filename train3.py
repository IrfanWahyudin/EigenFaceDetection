#!/usr/bin/env python

import matplotlib.pyplot as plt

import argparse
import imghdr
import numpy
import os
import random
import scipy.misc
import glob
import cv2
import numpy as np
# parser = argparse.ArgumentParser(description="Eigenface reconstruction demonstration")
# parser.add_argument("data",       metavar="DATA", type=str,   help="Data directory")
# parser.add_argument("n",          metavar="N",    type=int,   help="Number of training images", default=50)
# parser.add_argument("--variance",                 type=float, help="Desired proportion of variance", default = 0.95)

# arguments = parser.parse_args()

num_of_images = 20
dataDirectory    = 'dataset\\*'#arguments.data
numTrainingFaces = 15#arguments.n
variance         = 0.95#arguments.variance
gamma = []

faceDet = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faceDet_two = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
faceDet_three = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
faceDet_four = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")

if variance > 1.0:
  variance = 1.0
elif variance < 0.0:
  variance = 0.0


def extract_face(gray):
    #Detect face using 4 different classifiers
    face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face_two = faceDet_two.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face_three = faceDet_three.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face_four = faceDet_four.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)

    #Go over detected faces, stop at first detected face, return empty if no face.
    if len(face) == 1:
        facefeatures = face
    elif len(face_two) == 1:
        facefeatures = face_two
    elif len(face_three) == 1:
        facefeatures = face_three
    elif len(face_four) == 1:
        facefeatures = face_four
    else:
        facefeatures = ""
    #Cut and save face
    for (x, y, w, h) in facefeatures: #get coordinates and size of rectangle containing face

        gray = gray[y:y+h, x:x+w] #Cut the frame to size

    try:
        out = cv2.resize(gray, (350, 350)) #Resize face so all images have same size
        # cv2.imwrite("dataset\\%s\\%s.jpg" %(emotion, filenumber), out) #Write image
    except:
        pass #If error, pass file
    return out

def compute_average_face():
    global average_face
    global gamma
    files = glob.glob("dataset\\*" )
    M = len(files)
    
    for f in files:
        v = cv2.imread(f)
        gray = cv2.cvtColor(v, cv2.COLOR_BGR2GRAY)
        face = extract_face(gray)
        gamma.append(np.array(face, dtype='float64').ravel().tolist())


    gamma = np.array(gamma).T
    
    average_face = np.sum(gamma, axis=1) / num_of_images

    
    # cv2.imshow('average_face face', np.array(average_face,dtype='uint8').reshape(350, 350))
    # cv2.waitKey()
    return gamma


#
# Choose training images
#

trainingImages = list()

trainingImages = compute_average_face()
#
# Calculate & subtract average face
#

meanFace = numpy.zeros(trainingImages[0].shape)

for image in trainingImages:
  meanFace += 1/numTrainingFaces * image
  
cv2.imshow('average_face face', np.array(average_face,dtype='uint8').reshape(350, 350))
cv2.waitKey()
trainingImages = [ image - meanFace for image in trainingImages ] 
print(trainingImages)
#
# Calculate eigenvectors
#

x,y = trainingImages[0].shape
n   = x*y
A   = numpy.matrix( numpy.zeros((n,numTrainingFaces)) )

for i,image in enumerate(trainingImages):
  A[:,i] = numpy.reshape(image,(n,1))

M                         = A.transpose()*A
eigenvalues, eigenvectors = numpy.linalg.eig(M)
indices                   = eigenvalues.argsort()[::-1]
eigenvalues               = eigenvalues[indices]
eigenvectors              = eigenvectors[:,indices]

eigenvalueSum           = sum(eigenvalues)
partialSum              = 0.0
numEffectiveEigenvalues = 0

for index,eigenvalue in enumerate(eigenvalues):
  partialSum += eigenvalue
  if partialSum / eigenvalueSum >= variance:
    print("Reached", variance * 100, "%", "explained variance with", index+1 , "eigenvalues")
    numEffectiveEigenvalues = index+1
    break

V = numpy.matrix( numpy.zeros((n,numEffectiveEigenvalues)) )
for i in range(numEffectiveEigenvalues):
  V[:,i] = A*eigenvectors[:,i]

for i in range(numEffectiveEigenvalues):
 plt.imshow(V[:,i].reshape((x,y)),cmap=plt.cm.Greys_r)
 plt.show()

#
# Transform remaining images into "face space"
#

# remainingImages = list()

# for name in filenames:
#   if name not in trainingImageNames:
#     remainingImages.append( scipy.misc.imread(name) )

# remainingImages = [ image - meanFace for image in remainingImages ]

# for image in remainingImages:
#   weights = list()

#   for i in range(numEffectiveEigenvalues):
#     weights.append( (V[:,i].transpose() * image.reshape((n,1))).tolist()[0][0] )

#   reconstruction = numpy.matrix( numpy.zeros((n,1)) )
#   for i,w in enumerate(weights):
#     reconstruction += w*V[:,i]

#   f = plt.figure()
#   f.add_subplot(1, 2, 1)
#   plt.imshow(reconstruction.reshape((x,y)),cmap=plt.cm.Greys_r)
#   f.add_subplot(1, 2, 2)
#   plt.imshow(image.reshape((x,y)),cmap=plt.cm.Greys_r)
#   plt.show()