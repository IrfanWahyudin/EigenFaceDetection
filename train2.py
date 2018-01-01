#!/usr/bin/env python

__author__ = 'Aleksandar Gyorev'
__email__  = 'a.gyorev@jacobs-university.de'

import os
import cv2
import sys
import shutil
import random
import numpy as np
import glob
"""
A Python class that implements the Eigenfaces algorithm
for face recognition, using eigenvalue decomposition and
principle component analysis.

We use the AT&T data set, with 60% of the images as train
and the rest 40% as a test set, including 85% of the energy.

Additionally, we use a small set of celebrity images to
find the best AT&T matches to them.

Example Call:
    $> python2.7 eigenfaces.py att_faces celebrity_faces

Algorithm Reference:
    http://docs.opencv.org/modules/contrib/doc/facerec/facerec_tutorial.html
"""
class Eigenfaces(object):                                                       # *** COMMENTS ***

    faceDet = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faceDet_two = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
    faceDet_three = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
    faceDet_four = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")
    faces_count = 20

    faces_dir = 'dataset\\*'                                                             # directory path to the AT&T faces

    train_faces_count = 1                                                       # number of faces used for training
    test_faces_count = 5                                                      # number of faces used for testing

    l = train_faces_count * faces_count                                         # training images count
    m = 350                                                                      # number of columns of the image
    n = 350                                                                    # number of rows of the image
    mn = m * n                                                                  # length of the column vector

    def extract_face(self, gray):
        #Detect face using 4 different classifiers
        face = self.faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_two = self.faceDet_two.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_three = self.faceDet_three.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_four = self.faceDet_four.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)

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

    """
    Initializing the Eigenfaces model.
    """
    def __init__(self, _faces_dir = 'dataset\\*', _energy = 0.85):
        print('> Initializing started')

        self.faces_dir = _faces_dir
        self.energy = _energy
        self.training_ids = []                                                  # train image id's for every at&t face

        L = np.empty(shape=(self.mn, self.l), dtype='float64')                  # each row of L represents one train image
        cur_img = 0
        files = glob.glob("dataset\\*" )
        M = len(files)
    
        for f in files:
            v = cv2.imread(f)
            gray = cv2.cvtColor(v, cv2.COLOR_BGR2GRAY)
            face = self.extract_face(gray)
            img_col = np.array(face, dtype='float64').ravel().tolist()

            L[:, cur_img] = img_col[:]                                      # set the cur_img-th column to the current training image
            cur_img += 1

        self.mean_img_col = np.sum(L, axis=1) / self.l                          # get the mean of all images / over the rows of L

        print(self.mean_img_col)
        cv2.imshow('average_face face', np.array(self.mean_img_col,dtype='uint8').reshape(350, 350))
        cv2.waitKey()
        for j in range(0, self.l):                                             # subtract from all training images
            L[:, j] -= self.mean_img_col[:]

        C = np.matrix(L.transpose()) * np.matrix(L)                             # instead of computing the covariance matrix as
        C /= self.l                                                             # L*L^T, we set C = L^T*L, and end up with way
                                                                                # smaller and computentionally inexpensive one
                                                                                # we also need to divide by the number of training
                                                                                # images

        print('L.shape', L.shape)
        self.evalues, self.evectors = np.linalg.eig(C)                          # eigenvectors/values of the covariance matrix
        sort_indices = self.evalues.argsort()[::-1]                             # getting their correct order - decreasing
        self.evalues = self.evalues[sort_indices]                               # puttin the evalues in that order
        self.evectors = self.evectors[sort_indices]                             # same for the evectors

        evalues_sum = sum(self.evalues[:])                                      # include only the first k evectors/values so
        evalues_count = 0                                                       # that they include approx. 85% of the energy
        evalues_energy = 0.0
        for evalue in self.evalues:
            evalues_count += 1
            evalues_energy += evalue / evalues_sum

            if evalues_energy >= self.energy:
                break

        self.evalues = self.evalues[0:evalues_count]                            # reduce the number of eigenvectors/values to consider
        self.evectors = self.evectors[0:evalues_count]

        self.evectors = self.evectors.transpose()                               # change eigenvectors from rows to columns
        print('self.evectors.shape', self.evectors.shape)
        self.evectors = L * self.evectors                                       # left multiply to get the correct evectors
        norms = np.linalg.norm(self.evectors, axis=0)                           # find the norm of each eigenvector
        self.evectors = self.evectors / norms                                   # normalize all eigenvectors
        print('self.evectors.shape', self.evectors.shape)
        self.W = self.evectors.transpose() * L                                  # computing the weights
 
        print('> Initializing ended')

    """
    Classify an image to one of the eigenfaces.
    """
    def classify(self, path_to_img):
        img = cv2.imread(path_to_img, 0)                                        # read as a grayscale image
        img_col = np.array(img, dtype='float64').flatten()                      # flatten the image
        img_col -= self.mean_img_col                                            # subract the mean column
        img_col = np.reshape(img_col, (self.mn, 1))                             # from row vector to col vector

        S = self.evectors.transpose() * img_col                                 # projecting the normalized probe onto the
                                                                                # Eigenspace, to find out the weights
        print('W S Shape', self.W.shape, S.shape)                            
        diff = self.W - S                                                       # finding the min ||W_j - S||
        norms = np.linalg.norm(diff, axis=0)

        closest_face_id = np.argmin(norms)                                      # the id [0..240) of the minerror face to the sample
        return (closest_face_id / self.train_faces_count) + 1                   # return the faceid (1..40)

    """
    Evaluate the model using the 4 test faces left
    from every different face in the AT&T set.
    """
    def evaluate(self):
        print('> Evaluating AT&T faces started')
        results_file = os.path.join('results', 'att_results.txt')               # filename for writing the evaluating results in
        f = open(results_file, 'w')                                             # the actual file

        test_count = self.test_faces_count * self.faces_count                   # number of all AT&T test images/faces
        test_correct = 0
        for face_id in range(1, self.faces_count + 1):
            for test_id in range(1, 11):
                if (test_id in self.training_ids[face_id-1]) == False:          # we skip the image if it is part of the training set
                    path_to_img = os.path.join(self.faces_dir,
                            's' + str(face_id), str(test_id) + '.pgm')          # relative path

                    result_id = self.classify(path_to_img)
                    result = (result_id == face_id)

                    if result == True:
                        test_correct += 1
                        f.write('image: %s\nresult: correct\n\n' % path_to_img)
                    else:
                        f.write('image: %s\nresult: wrong, got %2d\n\n' %
                                (path_to_img, result_id))

        print('> Evaluating AT&T faces ended')
        self.accuracy = float(100. * test_correct / test_count)
        print('Correct: ' + str(self.accuracy) + '%')
        f.write('Correct: %.2f\n' % (self.accuracy))
        f.close()                                                               # closing the file

    """
    Evaluate the model for the small celebrity data set.
    Returning the top 5 matches within the AT&T set.
    Images should have the same size (92,112) and are
    located in the celebrity_dir folder.
    """
    def evaluate_celebrities(self, celebrity_dir='.'):
        print('> Evaluating celebrity matches started')
        for img_name in os.listdir(celebrity_dir):                              # go through all the celebrity images in the folder
            path_to_img = os.path.join(celebrity_dir, img_name)

            img = cv2.imread(path_to_img, 0)                                    # read as a grayscale image
            img_col = np.array(img, dtype='float64').flatten()                  # flatten the image
            img_col -= self.mean_img_col                                        # subract the mean column
            img_col = np.reshape(img_col, (self.mn, 1))                         # from row vector to col vector

            S = self.evectors.transpose() * img_col                             # projecting the normalized probe onto the
                                                                                # Eigenspace, to find out the weights

            diff = self.W - S                                                   # finding the min ||W_j - S||
            norms = np.linalg.norm(diff, axis=0)
            top5_ids = np.argpartition(norms, 5)[:5]                           # first five elements: indices of top 5 matches in AT&T set

            name_noext = os.path.splitext(img_name)[0]                          # the image file name without extension
            result_dir = os.path.join('results', name_noext)                    # path to the respective results folder
            os.makedirs(result_dir)                                             # make a results folder for the respective celebrity
            result_file = os.path.join(result_dir, 'results.txt')               # the file with the similarity value and id's

            f = open(result_file, 'w')                                          # open the results file for writing
            for top_id in top5_ids:
                face_id = (top_id / self.train_faces_count) + 1                 # getting the face_id of one of the closest matches
                subface_id = self.training_ids[face_id-1][top_id % self.train_faces_count]           # getting the exact subimage from the face

                path_to_img = os.path.join(self.faces_dir,
                        's' + str(face_id), str(subface_id) + '.pgm')           # relative path to the top5 face

                shutil.copyfile(path_to_img,                                    # copy the top face from source
                        os.path.join(result_dir, str(top_id) + '.pgm'))         # to destination

                f.write('id: %3d, score: %.6f\n' % (top_id, norms[top_id]))     # write the id and its score to the results file

            f.close()                                                           # close the results file
        print('> Evaluating celebrity matches ended')


# if __name__ == "__main__":
    # if len(sys.argv) < 2 or len(sys.argv) > 3:
    #     print('Usage: python2.7 eigenfaces.py ' \
    #         + '<att faces dir> [<celebrity faces dir>]')
    #     sys.exit(1)

if not os.path.exists('results'):                                           # create a folder where to store the results
    os.makedirs('results')
else:
    shutil.rmtree('results')                                                # clear everything in the results folder
    os.makedirs('results')

efaces = Eigenfaces()                                       # create the Eigenfaces object with the data dir

path_to_img = 'E:\\Projects\\Eigen Face\\cropped\\19.jpg'
ret = efaces.classify(path_to_img)                                                           # evaluate our model
print('closest:',  ret)
# if len(sys.argv) == 3:                                                      # if we have third argument (celebrity folder)
#     efaces.evaluate_celebrities(str(sys.argv[2]))                           # find best matches for the celebrities

# '''