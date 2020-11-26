#from keras.models import load_model
from mtcnn.mtcnn import MTCNN
import cv2
from PIL import Image
import numpy as np
from os import listdir
from matplotlib import pyplot
from os.path import isdir
from numpy import savez_compressed
# https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/

#model = load_model('facenet_keras.h5')
def extract_face(filename, required_size=(160, 160)):
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = np.asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    # bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    #image.show()
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array
 
# load the photo and extract the face
#pixels = extract_face('pic.jpg')
folder = '5-celebrity-faces-dataset/train/ben_afflek/'
'''
i = 1
for filename in listdir(folder):
    path = folder + filename
    face = extract_face(path)
    #print(i, face.shape)
    pyplot.subplot(2, 7, i)
    pyplot.axis('off')
    pyplot.imshow(face)
    i += 1
pyplot.show()
'''
def load_faces(directory):
    faces = list()
    for filename in listdir(directory):
        path = directory + filename
        face = extract_face(path)
        faces.append(face)
    return faces

def load_dataset(directory):
    X, y = list(), list()
    for subdir in listdir(directory):
        path = directory + subdir + '/'
        if not isdir(path):
            continue
        faces = load_faces(path)
        labels = [subdir for _ in range(len(faces))]
        print(labels)
        print('>loaded %d examples for class: %s' % (len(faces), subdir))
        X.extend(faces)
        y.extend(labels)
    return np.asarray(X), np.asarray(y)

trainX, trainy = load_dataset('5-celebrity-faces-dataset/train/')
testX, testy = load_dataset('5-celebrity-faces-dataset/val/')
savez_compressed('5-celebrity-faces-dataset.npz', trainX, trainy, testX, testy)












