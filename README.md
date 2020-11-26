# Face-Detection
Using FaceNet, MTCNN and SVM to detect face.

First, using MTCNN to detect the edge of face.
Then, use FaceNet to generate the face embeddings(features of the face).
Distances are calculated between different embeddings, and classify faces based on the distance.
Finally, test the model with test set.
Data is choosen from Kaggle, called: 5 celebrity faces dataset.
