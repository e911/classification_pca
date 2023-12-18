import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pca_faces import pca_eigenfaces


def load_orl_images(data_folder, subjects=40):
    images = []
    labels = []

    for subject_id in range(1, subjects + 1):
        subject_folder = os.path.join(data_folder, f"s{subject_id}")

        for i in range(1, 11):
            image_path = os.path.join(subject_folder, f"{i}.pgm")
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            img_flattened = img.flatten()

            images.append(img_flattened)
            labels.append(subject_id)

    return np.array(images), np.array(labels)


def split_dataset(images, labels, train_subjects=40, face_recognition=True):
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    unique_labels = np.unique(labels)

    for label in unique_labels[:train_subjects]:
        label_indices = np.where(labels == label)[0]
        if label <= 35:
            train_indices, test_indices = train_test_split(label_indices, test_size=2 / 10.0, random_state=42)

            train_images.extend(images[train_indices])

            test_images.extend(images[test_indices])

            if face_recognition:
                train_labels.extend([1] * len(labels[train_indices]))
                test_labels.extend([1] * len(labels[test_indices]))
            else:
                train_labels.extend(labels[train_indices])
                test_labels.extend(labels[test_indices])
        else:
            test_images.extend(images[label_indices])
            if face_recognition:
                test_labels.extend([1] * len(labels[label_indices]))
            else:
                test_labels.extend(labels[label_indices])
    return np.array(train_images), np.array(train_labels), np.array(test_images), np.array(test_labels)


def get_p_order_data(data, order):
    result = []

    for x in data:
        r = []
        for p in range(1, order + 1):
            r.extend(x ** p)

        result.append(r)
    return np.array(result)


def add_bias(data):
    return np.insert(data, 0, 1.0, axis=1)


def x_data_preprocess(x_data, order):
    x_feature = get_p_order_data(x_data, order)

    x_feature = add_bias(x_feature)
    x_feature = x_feature.T
    return x_feature


def project_data(data, eigenvectors, mean_X, n_pcs):
    centered_data = data - mean_X
    coefficients = np.dot(centered_data, eigenvectors[:, :n_pcs])
    return coefficients


if __name__ == "__main__":
    data_folder = os.getcwd() + "/att_faces"

    images, labels = load_orl_images(data_folder, 40)
    train_data, train_labels, test_data, test_labels = split_dataset(images, labels, 40, True)

    # scaler = StandardScaler()
    # Z_train = scaler.fit_transform(train_data)
    # Z_test = scaler.fit_transform(test_data)
    #
    # Z_train = x_data_preprocess(Z_train, 1)
    # Z_test = x_data_preprocess(Z_test, 1)

    eigenvalues, eigenvectors, mean_face, _, n_pcs_90 = pca_eigenfaces(train_data)
    Z_train = project_data(train_data, eigenvectors, mean_face, n_pcs_90).T
    Z_test = project_data(test_data, eigenvectors, mean_face, n_pcs_90).T

    nclass = len(np.unique(train_labels))
    ntrain = len(train_labels)

    Y_train = np.zeros((nclass, ntrain))
    for i in range(ntrain):
        Y_train[int(train_labels[i]) - 1, i] = 1
    W = np.dot(Y_train, np.dot(Z_train.T, np.linalg.inv(np.dot(Z_train, Z_train.T))))

    Y_pred = np.dot(W, Z_test)

    predicted_labels = np.argmax(Y_pred, axis=0) + 1
    print("Face recognition for test data sets:")
    print(predicted_labels)
    accuracy = np.sum(predicted_labels == test_labels) / len(test_labels) * 100
    print(f'Accuracy: {accuracy:.2f}%')
    # print('Weight Matrix W:')
    # print(W)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(train_data[:, 0], train_data[:, 1], c=train_labels, marker='o', label='Training Data')
    ax.scatter(test_data[:, 0], test_data[:, 1], c=predicted_labels, marker='x', label='Test Data (Predicted)')

    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Scatter Matrix with Training and Test Data')
    ax.legend()

    plt.show()

    train_data, train_labels, test_data, test_labels = split_dataset(images, labels, 40, False)

    scaler = StandardScaler()
    Z_train = scaler.fit_transform(train_data)
    Z_test = scaler.fit_transform(test_data)

    Z_train = x_data_preprocess(Z_train, 1)
    Z_test = x_data_preprocess(Z_test, 1)

    nclass = len(np.unique(train_labels))
    ntrain = len(train_labels)

    Y_train = np.zeros((nclass, ntrain))
    for i in range(ntrain):
        Y_train[int(train_labels[i]) - 1, i] = 1
    W = np.dot(Y_train, np.dot(Z_train.T, np.linalg.inv(np.dot(Z_train, Z_train.T))))

    Y_pred = np.dot(W, Z_test)

    predicted_labels = np.argmax(Y_pred, axis=0) + 1
    print("Face detection for test data sets:")
    print(predicted_labels)
    accuracy = np.sum(predicted_labels == test_labels) / len(test_labels) * 100
    print(f'Accuracy: {accuracy:.2f}%')
    # print('Weight Matrix W:')
    # print(W)
