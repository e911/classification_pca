import numpy as np
import torch
import os
import cv2
import random
from PIL import Image
from pca_faces import pca_eigenfaces
from classification_faces import project_data
import time

def get_pca_data(train_data,test_data):
    print('Generating data using PCA')
    eigenvalues, eigenvectors, mean_face, _, n_pcs_90 = pca_eigenfaces(train_data)
    Z_train = project_data(train_data, eigenvectors, mean_face, n_pcs_90)
    Z_test = project_data(test_data, eigenvectors, mean_face, n_pcs_90)
    return Z_train,Z_test


def get_lda_data(train_data,train_label,test_data):
    print('Generating data using LDA')
    eigenvectors, train_means = lda(train_data, train_label)
    # Transform the test data using the LDA eigenvectors and training means
    X_test_centered = test_data - train_means
    X_train_centered = train_data - train_means
    X_test_lda = X_test_centered.dot(eigenvectors)
    X_train_lda = X_train_centered.dot(eigenvectors)
    return X_train_lda,X_test_lda

def load_orl_images(data_folder):
    images = []
    labels = []
    subject_folders = [f.path for f in os.scandir(data_folder) if f.is_dir()]

    for subject_folder in subject_folders:
        subject_label = os.path.basename(subject_folder)[1:]

        for file in os.scandir(subject_folder):
            if file.name.endswith(".pgm") and file.is_file():
                image_path = file.path
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                img_flattened = img.flatten()
                images.append(img_flattened)
                labels.append(subject_label)
    return np.array(images), np.array(labels)

def load_and_process_images(directory, num_images=400):
    # List all JPEG files in the directory
    all_images = [file for file in os.listdir(directory) if file.endswith('.jpg')]
    random.seed(42)
    selected_images = random.sample(all_images, num_images)
    
    processed_images = []
    processed_images = np.empty((num_images,112*92))
    for i,img_name in enumerate(selected_images):
        img_path = os.path.join(directory, img_name)
        img = Image.open(img_path)
        img_gray = img.convert('L')
        img_resized = img_gray.resize((112, 92))
        processed_images[i]= np.array(img_resized).flatten()

    return processed_images


def lda(X, y, num_components=None):
    if(num_components==None):
        num_components=y.shape[1]-1
        print(num_components)
    y=[np.where(y[i]==1)[0][0] for i in range(len(y))]
    class_labels = np.unique(y)
    n_features = X.shape[1]
    mean_overall = np.mean(X, axis=0)
    S_W = np.zeros((n_features, n_features))
    S_B = np.zeros((n_features, n_features))

    for c in class_labels:
        X_c = X[y == c]
        mean_vec = np.mean(X_c, axis=0)
        S_W += np.dot((X_c - mean_vec).T, (X_c - mean_vec))

        n_c = X_c.shape[0]
        mean_diff = (mean_vec - mean_overall).reshape(n_features, 1)
        S_B += n_c * (mean_diff).dot(mean_diff.T)
    print('calculating inverse')
    start= time.time()
    A = np.linalg.inv(S_W).dot(S_B)
    end = time.time()
    start= time.time()
    eigenvalues, eigenvectors = np.linalg.eig(A)

    # Sort the eigenvectors by decreasing eigenvalues
    eigenvectors = eigenvectors[:, np.argsort(-eigenvalues)]
    return eigenvectors[:, :num_components], mean_overall


def load_train_test_data(data_folder='./att_faces',task='identification'):
    images,labels= load_orl_images(data_folder)
    indices = [x for x in range(images.shape[0])]
    test_images_indices = indices[8:350:10] + indices[9:350:10] + indices[350:]
    train_images_indices = list(set(indices).difference(set(test_images_indices)))
    test_images= images[test_images_indices]
    train_images=  images[train_images_indices]
    train_images = np.array(train_images)
    test_images = np.array(test_images)
    train_labels= None
    test_labels= None
    ndata = len(images)
    np.random.seed(42)

    if(task=='identification'):
        nclass = len(np.unique(labels))
        Y = np.zeros((ndata,nclass))
        for i in range(ndata):
            Y[i,int(labels[i]) - 1] = 1
        train_labels= Y[train_images_indices]
        test_labels=  Y[test_images_indices]
    elif(task=='recognition'):
        Y = np.zeros((ndata*2,2))
        Y[:ndata,1]=1
        Y[ndata:,0]=1
        negative_train_indices = [x+400 for x in train_images_indices]
        negative_test_indices=   [x+400 for x in test_images_indices]
        train_labels= np.concatenate((Y[train_images_indices,:],Y[negative_train_indices,:]))
        test_labels=  np.concatenate((Y[test_images_indices,:],Y[negative_test_indices,:])) 
        negatives= load_and_process_images("./non_face", num_images=400)
        train_images = np.concatenate((train_images,negatives[train_images_indices]))
        test_images = np.concatenate((test_images,negatives[test_images_indices]))
    shuffled_indices = np.random.permutation(len(train_images))
    shuffled_train_data = train_images[shuffled_indices]
    shuffled_train_labels = train_labels[shuffled_indices]
    shuffled_indices = np.random.permutation(len(test_images))
    shuffled_test_data = test_images[shuffled_indices]
    shuffled_test_labels = test_labels[shuffled_indices]
    print("train images",shuffled_train_data.shape)
    print("train labels",shuffled_train_labels.shape)
    print("test images",shuffled_test_data.shape)
    print("test labels",shuffled_test_labels.shape)

    return shuffled_train_data,shuffled_train_labels,shuffled_test_data,shuffled_test_labels


    


if __name__=="__main__":

    print("Start")
    train_data, train_label, val_data, test_label = load_train_test_data("./att_faces",'identification')
    train_arr,test_data_arr = get_pca_data(train_data,val_data)
    print(train_arr.shape)
    print(train_arr.dtype)
    print(test_data_arr.shape)
    print(test_data_arr.dtype)

    #print(imgs.shape)
    
    