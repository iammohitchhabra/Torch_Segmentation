import numpy as np
import h5py
import sys
import os
import cv2

directory="./test/"
desired_width=240
desired_height=180
#Give the path of the textfile, read the file line by line and return the list of lines.
def readfile(textfile,listoflines):
	with open(textfile) as file:
		for line in file:

			listoflines.append(line[:-1])	
	return listoflines

def load_image(img_path):
	img=cv2.imread(img_path,1)
	img = cv2.resize(img,(desired_width, desired_height), interpolation = cv2.INTER_CUBIC)
	return img


names=[]
names=readfile('./test/filenames.txt',names)
l=len(names)

testfiles=[]
outputfiles=[]
out=[]
test=[]
for i in xrange(l):
		testfiles.append(directory+names[i])
		test.append(names[i])
files=[]
l=len(testfiles)
for i in xrange(l):
	files.append(testfiles[i][:-4])

#create a new hdf5 file
f=h5py.File("./test/testset.hdf5","w")


for i in xrange(l):
	img=load_image(testfiles[i])
	#convert the image to float using minmax normalization
	img = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
	#f.create_dataset("/dataset/"+train[i][:-4]+"/"+"train",data=img)
	f.create_dataset("/dataset/test/"+str(i),data=img)

f.close()

	