
#!/usr/bin/env pythonb
import cv2
import glob
import random
import numpy as np
import glob
from PIL import Image
import os
import matplotlib.pyplot as plt
faceDet = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")			# Front facing face pattern #1
faceDet_two = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")			# Front facing face pattern #2
faceDet_three = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")		# Front facing face pattern #3
faceDet_four = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")	# Front facing face pattern #4

num_of_images = 24												# Define number of images used for training process
average_face = None 											# Average/Mean Face
A = []															# Image Repository
phi = []														# Variable to handle faces that substracted by the Mean Face
C = None	
u	=   []														# Covariance Matrix
w, v = None, None												# w = Eigen Values, v = Eigen Vectors
num_of_v = 0													# Number of The Most Optimum Eigen Vectors
best_of_v = []													# The Best Eigen Vectors
weights = []													# Image Multiplier used by New Image to adapt with the precomputed Eigen Faces 							
eigenfaces = None												# The Eigen Faces
image_size = 350*350											# Image size for training 
threshold = 1.00												# Threshold to define how many eigen values/vectors will be used						
class_table = ['anna','anna','anna','anna',\
				'charles','charles','charles','charles',\
				'irfan','irfan','irfan','irfan',\
				'jessie','jessie','jessie','jessie',\
				'linda','linda','linda','linda',\
				'maria','maria','maria','maria']				# The Class Table

def extract_face(gray):
	#Detect face using 4 different classifiers
	x = 0
	y = 0 
	w = 0
	h = 0
	out = None
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
	outs = []
	xywh = []
	for (x, y, w, h) in facefeatures: #get coordinates and size of rectangle containing face
		gray = gray[y:y+h, x:x+w] #Cut the frame to size
		gray = cv2.resize(gray, (350, 350))
		outs.append(gray)
		xywh.append([x,y,w,h])

	print('number of faces', len(outs))
	try:
	    out = cv2.resize(gray, (350, 350)) #Resize face so all images have same size
	    # cv2.imwrite("dataset\\%s.jpg" %(i), out) #Write image
	except:
		pass #If error, pass file
	return xywh, outs


'''
Here we extract the Average/Mean Face of given images
This method reads all the images from give path, and stores them into global A variable
The very end of this method will returns a 122500*1 or 350*350 mean face that will be 
used to susbtract each of the sample image
'''
def compute_average_face():
	global average_face
	global A
	files = sorted(glob.glob("cropped\\*"))
	print(files)
	M = len(files)
	
	for i, f in enumerate(files):
		v = cv2.imread(f)
		print(f)
		gray = cv2.cvtColor(v, cv2.COLOR_BGR2GRAY)

		# x, y, w, h, face = extract_face(i, gray)
		face = gray
		A.append(np.array(face, dtype='float64').ravel().tolist())


	A = np.array(A).T
	average_face = np.zeros(A.T[0].shape)

	for g in A.T:
		average_face += 1/num_of_images * g

	
	# cv2.imshow('average_face face', np.array(average_face,dtype='int8').reshape(350, 350))
	# cv2.waitKey()

'''
After we have the average image, then we use it to substract each of training image  
'''
def substract_mean_face():
	global phi
	global A
	global average_face

	phi = np.empty([image_size, num_of_images])

	for i in range(0, num_of_images):
		phi[:, i] = A[:, i] - average_face

'''
Now, we have the substracted images, and we can compute the Covariance Matrix
C = phi^T * phi
'''
def compute_covariance():
	global phi
	global C

	C = np.matrix(phi.transpose()) * np.matrix(phi) 
	C /= num_of_images
	

	# phi = np.array(phi, dtype='uint8')
	# phi_transpose = phi.T
	# C = np.dot(phi_transpose, phi)


'''
Now compute the Eigenvectors
Select The Best Eigen Vectors
'''
def compute_eigenvectors():
	global C
	global num_of_v
	global w, v, threshold
	eigen_count = 0
	eigen_sum = 0.0
	size_of_C = C.shape[0]
	w, v = np.linalg.eig(C)


	for i in range(len(w)):
		eigen_count += 1
		eigen_sum += np.sum(w[0:i]/np.sum(w[0:size_of_C]))

		if eigen_sum > threshold:
			break

	num_of_v = eigen_count

	print('Best eigen vectors: ', num_of_v)
	

'''
Now compute the Eigenvectors
Select The K Best Eigen Vectors
The Eigen Vectors is simply n*n square matrix
We select the only K Best Eigen Vectors that represent % of threshold 
'''
def keep_best_eigenvectors():
	global num_of_v, best_of_v, w, v
	eigen_count = 0
	sort_indices = w.argsort()[::-1] 
	w = w[sort_indices]

	best_of_v = v[sort_indices]
	# best_of_v = best_of_v.T


'''
Obtain the Eigen Faces from multiplying phi with the selected Eigen Vectors
We can visualize the eigen faces, and we will see some weird faces that represent the key identifiers of each faces
like shape of eyes, nose, lips etc.
'''
def extract_eigen_faces():
	global phi, eigenfaces, u, best_of_v

	eigenfaces = np.matrix(np.zeros((image_size, num_of_v)))
	eigenfaces = phi  * best_of_v



	eigenfaces = np.array(eigenfaces[:,0:num_of_v])
	for ef in eigenfaces.T:
		cv2.imshow('eigenface',np.array(ef, dtype='uint8').reshape(350, 350))
		cv2.waitKey()


def extract_weights(idx):
	global  A,best_of_v, weights, phi, u, eigenfaces

	norms = np.linalg.norm(eigenfaces, axis=0)	
	weights = []
	# for i in range(num_of_v):
	# 	img = phi[:, idx].reshape(image_size, 1)
	# 	eigenface_reshape = eigenfaces[:,i].transpose()
	# 	# print((np.matrix(eigenface_reshape) * np.matrix(img)).tolist())
	# 	weights.append((np.matrix(eigenface_reshape) * np.matrix(img)).tolist()[0][0])
	weights = np.dot(np.matrix(eigenfaces).T, np.matrix(phi[:,idx]).T)
	print('w shape', weights.shape)
	return weights

def construct_old_face(idx):
	global u, average_face, phi, weights, eigenfaces 	
	_weights = extract_weights(idx)

	print(_weights)
	recon = np.matrix( np.zeros((1, image_size)) )
	
	i = 0

	norms = np.linalg.norm(eigenfaces, axis=0)
	eigenfaces = eigenfaces / norms
	for i, w in enumerate(weights):
		
		curr_eigenface = eigenfaces[:,i]
		
		recon = recon + (w * curr_eigenface.reshape(1, image_size))
		plt.imshow(np.array((recon), dtype='uint16').reshape(350,350),cmap=plt.cm.Greys_r)
		plt.show()

def classify(face):
	global A, eigenfaces, phi, average_face

	local_weights = np.matrix(eigenfaces).T * np.matrix(phi)
	
	# img_col = A[:, 23]
	# face = face.flatten()
	# face2 = cv2.imread('E:\\Projects\\Eigen Face\\test\\21.jpg')
	face2 = face
	face2 = np.array(face2, dtype='float64').ravel()
	img_col = face2

	# from scipy import spatial
	# dataSetI = img_col
	# dataSetII = face
	# result = 1 - spatial.distance.cosine(dataSetI, dataSetII)
	# print(result)
	print('img_col.shape', img_col.shape)
	print('face.shape', face.shape)


	f = open('face', 'w')
	f.write(str(face2.tolist()))
	f.close()


	# img_col = cv2.resize(face, (350, 350))
	
	
	img_col = img_col - average_face
	img_col = img_col.reshape(image_size, 1)

	S = np.matrix(eigenfaces).T * np.matrix(img_col)

	print('S Shape', S.shape)
	diff = local_weights - S

	norms = np.linalg.norm(diff, axis=0)

	# print(norms)
	closest_face_id = np.argmin(norms)  

	class_name = class_table[closest_face_id]
	return class_name + ' -- ' + str(closest_face_id)


if __name__ == '__main__':
	print('Compute Average Face...')
	compute_average_face()
	print('Substract Mean Face...')
	substract_mean_face()
	print('Compute Covariance...')
	compute_covariance()
	print('Compute Eigen Vectors...')
	compute_eigenvectors()
	print('Keep Best of Eigen Vectors...')
	keep_best_eigenvectors()
	print('Extract Eigen Faces...')
	extract_eigen_faces()
	print('Construct Old Face...')
	# for j in range(num_of_images):
	construct_old_face(20)

	print('Classify...')

	cap = cv2.VideoCapture(0)

	while(True):
	    # Capture frame-by-frame
	    ret, frame = cap.read()

	    # Our operations on the frame come here
	    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	    coords, faces = extract_face(gray)
	    coords_faces = zip(coords, faces)
	    for cf in coords_faces:
	    	x = cf[0][0]
	    	y = cf[0][1]
	    	w = cf[0][2]
	    	h = cf[0][3]
	    	face = cf[1]
	    	print('face.shape', face.shape)

	    	print(x,y,w,h)
	    	class_name = classify(face)
	    	cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)

	    	font = cv2.FONT_HERSHEY_SIMPLEX
	    	cv2.putText(frame,class_name,(x,y), font, 1,(0,255,0),2,cv2.LINE_AA)

	    cv2.imshow('frame',frame)
	    if cv2.waitKey(1) & 0xFF == ord('q'):
		    break

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()



