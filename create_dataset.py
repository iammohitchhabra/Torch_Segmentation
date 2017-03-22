import numpy as np
import h5py
import sys
import os
import cv2

directory="/home/naruto/Documents/datasets/camvid/CamSeq01/"
desired_width=240
desired_height=180

#Give the path of the textfile, read the file line by line and return the list of lines.
def readfile(textfile,listoflines):
	with open(textfile) as file:
		for line in file:

			listoflines.append(line[:-1])	
	return listoflines

def load_image(img_path,desired_width,desired_height):
	img=cv2.imread(img_path,1)
	img = cv2.resize(img,(desired_width, desired_height), interpolation = cv2.INTER_CUBIC)
	return img


names=[]
names=readfile('/home/naruto/Documents/torch/experiments/segmentation/filenames.txt',names)
l=len(names)

trainfiles=[]
outputfiles=[]
out=[]
train=[]
for i in xrange(l):
	if i%2==1:
		trainfiles.append(directory+names[i])
		train.append(names[i])
	else:
		outputfiles.append(directory+names[i])
		out.append(names[i])
files=[]
l=len(trainfiles)
for i in xrange(l):
	files.append(trainfiles[i][:-4])

#create a new hdf5 file
f=h5py.File("/home/naruto/Documents/torch/experiments/segmentation/dataset.hdf5","w")


for i in xrange(l):
	img=load_image(trainfiles[i],desired_width,desired_height)
	#convert the image to float using minmax normalization
	img = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
	#f.create_dataset("/dataset/"+train[i][:-4]+"/"+"train",data=img)
	f.create_dataset("/dataset/train/"+str(i),data=img)
	img=load_image(outputfiles[i],desired_width,desired_height)
	img2=np.zeros((desired_height,desired_width))
	for m in xrange(desired_height):
		for n in xrange(desired_width):
			#Label the pixels corresponding to roads 1 and all else to 0.
			if img[m][n][0]==128 and img[m][n][1]==64 and img[m][n][2]==128:
				img2[m][n]=18
			elif img[m][n][0]==64 and img[m][n][1]==128 and img[m][n][2]==64: 
				img2[m][n]=1
			elif img[m][n][0]==128 and img[m][n][1]==0 and img[m][n][2]==192: 
				img2[m][n]=2
			elif img[m][n][0]==192 and img[m][n][1]==128 and img[m][n][2]==0: 
				img2[m][n]=3
			elif img[m][n][0]==64 and img[m][n][1]==128 and img[m][n][2]==0: 
				img2[m][n]=4
			elif img[m][n][0]==0 and img[m][n][1]==128 and img[m][n][2]==128: 
				img2[m][n]=5
			elif img[m][n][0]==192 and img[m][n][1]==0 and img[m][n][2]==64: 
				img2[m][n]=6
			elif img[m][n][0]==192 and img[m][n][1]==0 and img[m][n][2]==64: 
				img2[m][n]=7
			elif img[m][n][0]==64 and img[m][n][1]==128 and img[m][n][2]==192: 
				img2[m][n]=8
			elif img[m][n][0]==128 and img[m][n][1]==192 and img[m][n][2]==192: 
				img2[m][n]=9
			elif img[m][n][0]==64 and img[m][n][1]==64 and img[m][n][2]==128: 
				img2[m][n]=10
			elif img[m][n][0]==192 and img[m][n][1]==0 and img[m][n][2]==128: 
				img2[m][n]=11
			elif img[m][n][0]==64 and img[m][n][1]==0 and img[m][n][2]==192: 
				img2[m][n]=12
			elif img[m][n][0]==64 and img[m][n][1]==128 and img[m][n][2]==128: 
				img2[m][n]=13
			elif img[m][n][0]==192 and img[m][n][1]==0 and img[m][n][2]==192: 
				img2[m][n]=14
			elif img[m][n][0]==64 and img[m][n][1]==64 and img[m][n][2]==128: 
				img2[m][n]=15
			elif img[m][n][0]==128 and img[m][n][1]==192 and img[m][n][2]==64: 
				img2[m][n]=16
			elif img[m][n][0]==0 and img[m][n][1]==64 and img[m][n][2]==64: 
				img2[m][n]=17
			elif img[m][n][0]==192 and img[m][n][1]==128 and img[m][n][2]==128: 
				img2[m][n]=19
			elif img[m][n][0]==192 and img[m][n][1]==0 and img[m][n][2]==0: 
				img2[m][n]=20
			elif img[m][n][0]==128 and img[m][n][1]==128 and img[m][n][2]==192: 
				img2[m][n]=21
			elif img[m][n][0]==128 and img[m][n][1]==128 and img[m][n][2]==128: 
				img2[m][n]=22
			elif img[m][n][0]==192 and img[m][n][1]==128 and img[m][n][2]==64: 
				img2[m][n]=23
			elif img[m][n][0]==64 and img[m][n][1]==0 and img[m][n][2]==0: 
				img2[m][n]=24
			elif img[m][n][0]==64 and img[m][n][1]==64 and img[m][n][2]==0: 
				img2[m][n]=25
			elif img[m][n][0]==128 and img[m][n][1]==64 and img[m][n][2]==192: 
				img2[m][n]=26
			elif img[m][n][0]==0 and img[m][n][1]==128 and img[m][n][2]==128: 
				img2[m][n]=27
			elif img[m][n][0]==192 and img[m][n][1]==128 and img[m][n][2]==192: 
				img2[m][n]=28
			elif img[m][n][0]==64 and img[m][n][1]==0 and img[m][n][2]==64: 
				img2[m][n]=29
			elif img[m][n][0]==0 and img[m][n][1]==192 and img[m][n][2]==192: 
				img2[m][n]=30
			elif img[m][n][0]==0 and img[m][n][1]==192 and img[m][n][2]==64: 
				img2[m][n]=32
			else:  
				img2[m][n]=31
	
	f.create_dataset("/dataset/label/"+str(i),data=img2)
f.close()

	