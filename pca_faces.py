import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Function to load and preprocess images from the ORL database
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


def pca_eigenfaces(images):
    mean_X = np.mean(images, axis=0)
    centered_X = images - mean_X

    covariance_matrix = np.cov(centered_X, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    explained_variance_ratio = sorted_eigenvalues / np.sum(sorted_eigenvalues)
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    n_pcs_90 = np.argmax(cumulative_variance_ratio > 0.90) + 1
    print(f"Number of PCs for 90% variance: {n_pcs_90}")

    plt.plot(cumulative_variance_ratio, marker='o')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Cumulative Variance Ratio vs. Number of PCs')
    plt.grid(True)
    plt.show()

    return sorted_eigenvalues, sorted_eigenvectors, mean_X, centered_X, n_pcs_90


def visualize_eigenfaces(eigenvectors, num_faces=8):
    fig, axes = plt.subplots(1, num_faces, figsize=(12, 3))
    for i in range(num_faces):
        eigenface = np.real(eigenvectors[:, i]).reshape(112,92)
        axes[i].imshow(eigenface, cmap='gray')
        axes[i].set_title(f'Eigenface {i + 1}')
    plt.show()


def reconstruct_image(sorted_eigenvectors, centered_X, mean_X, n_pcs_90):
    example_image = centered_X[0, :]
    coefficients = np.dot(example_image, sorted_eigenvectors[:, :n_pcs_90])
    reconstructed_image = mean_X + np.dot(coefficients, sorted_eigenvectors[:, :n_pcs_90].T)
    reconstructed_image = np.clip(reconstructed_image, 0, 255)  # Clip to valid pixel values
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(example_image.reshape(112,92), cmap='gray')
    axes[0].set_title('Original Image')
    axes[1].imshow(np.real(reconstructed_image.reshape(112,92)), cmap='gray')
    axes[1].set_title(f'Reconstructed Image using {n_pcs_90} PCs')
    plt.show()

if __name__ == "__main__":
    data_folder = os.getcwd() + "/att_faces"

    images, labels = load_orl_images(data_folder)

    eigenvalues, eigenvectors, mean_face, centered_face, n_pcs_90 = pca_eigenfaces(images)
    print(eigenvalues, eigenvectors, mean_face)
    visualize_eigenfaces(eigenvectors)
    for num_faces in range(1, 11):
        reconstruct_image(eigenvectors, centered_face, mean_face, num_faces)

    reconstruct_image(eigenvectors, centered_face, mean_face, n_pcs_90)
