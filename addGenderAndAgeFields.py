import cv2
import time
import numpy as np
import json
from urllib.request import urlopen
import urllib
import sys

# Dependencies
# pip install opencv-python
# or
# git clone https://github.com/opencv/opencv.git

# is recommended to install it in new enviroment to avoid instalation errors

# The folfer "data" must be to same level where the script is executed



# This script add gender and age fields to each json node list processing your url image which key must be specified, if the image do not contain faces your node is not included

# To execute the script is necessary to specify the input json name file, your field node key of image choose to process and the output json name file 
# python addGenderAndAgeFields.py inputJsonFilePath fieldImageJsonKey outputJsonFilePath 

# Example
# python addGenderAndAgeFields.py lastoutputjson.json publicationImageUrl outPutDatass.json

# You can change the input json file by other generated in the phase one

# origin documentation
# https://stackabuse.com/reading-and-writing-json-to-a-file-in-python/
# https://github.com/raunaqness/Gender-and-Age-Detection-OpenCV-Caffe
# https://github.com/sebastian-lapuschkin/understanding-age-gender-deep-learning-models

# The results are sometimes not successful


# get a cv2 image object from a image web url specified
def getCv2ImageFromUrl(imageUrl):

	response = urllib.request.urlopen(imageUrl)
	image = np.asarray(bytearray(response.read()), dtype="uint8")
	
	return cv2.imdecode(image, cv2.IMREAD_COLOR)

	
# get all faces objects with your coordinates from a cv2 image object specified
def getFacesFromCv2Image(cv2Image):

	face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')
	gray = cv2.cvtColor(cv2Image, cv2.COLOR_BGR2GRAY)
	
	return face_cascade.detectMultiScale(gray, 1.1, 5)	
	
	
# get the cv2 blob object of an image thorugh your coordinates
def getBlobFaceFromCoordinates(x, y, w, h, image):

	MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
	cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 2)
	face_img = image[y:y+h, h:h+w].copy()
	
	return cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

	
# get the gender of a cv2 face blob object specified
def getGenderFromCV2BlobFace(blob):

	gender_list = ['Male', 'Female']
	
	gender_net = cv2.dnn.readNetFromCaffe(
		'data/deploy_gender.prototxt', 
		'data/gender_net.caffemodel')
		
	gender_net.setInput(blob)
	gender_preds = gender_net.forward()
	
	return gender_list[gender_preds[0].argmax()]
	
	
	
# get the age of the cv2 face blob object specified
def getAgeFromCv2BlobFace(blob):

	age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
	
	age_net = cv2.dnn.readNetFromCaffe(
		'data/deploy_age.prototxt', 
		'data/age_net.caffemodel')
		
	age_net.setInput(blob)
	age_preds = age_net.forward()
	age = age_list[age_preds[0].argmax()]
			
	listRangeAge = age.replace("(", "").replace(")", "").split(",")
	
	return int((int(listRangeAge[0]) + int(listRangeAge[1])) / 2.0)
	
	
# get the json of a file path specified
def getJsonFromFile(filePath):

	with open(filePath) as f:
		data = json.load(f)["0"]
		
	return data

	
# save the content json in the file path specified
def saveJsonFile(filePath, jsonContent):

    with open(filePath, 'w') as outFile:
        json.dump(jsonContent, outFile)

		

# adding your age and gender for each json nodes 
def addFeatures(data, imageKeyFieldToProcess):

	# initializing the output json
	outputJson = {}
	outputJson["0"] = []
		
	print("%s json nodes found" % str(len(data)))

			
	for jsonUser in data:
	
		# getting the cv2 object image
		imageUrl = jsonUser[imageKeyFieldToProcess]
		image = getCv2ImageFromUrl(imageUrl)

		# getting the faces of the image
		faces = getFacesFromCv2Image(image)		
		
		if(len(faces)>0):
			print("Found {} faces".format(str(len(faces))))
		else:
			print("No faces found")

		for (x, y, w, h )in faces:
			
			# getting the blob of the face
			blob = getBlobFaceFromCoordinates(x, y, w, h, image)
			
			try:
				# getting the gender
				gender = getGenderFromCV2BlobFace(blob)
				print("Gender : " + gender)
				jsonUser["gender"] = gender

				# getting the age
				age = getAgeFromCv2BlobFace(blob)
				print("Age: %s" % age)
				jsonUser["age"] = age

				# adding the json node to the output json
				outputJson["0"].append(jsonUser)
			except Exception as e:
				print("Exception processing an image. %s" % str(e))
				
	print(outputJson)
	
	return outputJson
	
	

# init function 
def __init__():

	# getting the arguments/parameters
	inputFilePath = sys.argv[1]
	imageKeyFieldToProcess = sys.argv[2]
	outputFilePath = sys.argv[3]
	
	data = getJsonFromFile(inputFilePath)

	outputJson = addFeatures(data, imageKeyFieldToProcess)
	
	saveJsonFile(outputFilePath, outputJson)


	
__init__()