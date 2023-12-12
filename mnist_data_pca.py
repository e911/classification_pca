import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

mnist = fetch_openml("mnist_784")
x = mnist.data.astype('float64')[0:2001]
y = mnist.target.astype('int64')[0:2001]
digit_4_indices = y
X = x.to_numpy().astype('float64')

mean_X = np.mean(X, axis=0)
centered_X = X - mean_X

covariance_matrix = np.cov(centered_X, rowvar=False)

eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

explained_variance_ratio = sorted_eigenvalues / np.sum(sorted_eigenvalues)
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

plt.plot(cumulative_variance_ratio, marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Cumulative Variance Ratio vs. Number of PCs')
plt.grid(True)
plt.show()

n_pcs_90 = np.argmax(cumulative_variance_ratio > 0.90) + 1
print(f"Number of PCs for 90% variance: {n_pcs_90}")


mean_of_dataset = np.mean(X, axis=0)
plt.imshow(np.real(mean_of_dataset.reshape(28, 28)), cmap='gray')
plt.title('Mean Image of Digit 4')


n_components_to_visualize = 5
fig, axes = plt.subplots(1, n_components_to_visualize, figsize=(15, 2))
for i in range(n_components_to_visualize):
    pc_image = np.real(sorted_eigenvectors[:, i]).reshape(28, 28)
    axes[i].imshow(pc_image, cmap='gray')
    axes[i].set_title(f'PC {i+1}')
plt.show()



example_image = centered_X[0, :]
coefficients = np.dot(example_image, sorted_eigenvectors[:, :n_pcs_90])
reconstructed_image = mean_X + np.dot(coefficients, sorted_eigenvectors[:, :n_pcs_90].T)
reconstructed_image = np.clip(reconstructed_image, 0, 255)  # Clip to valid pixel values
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(example_image.reshape(28, 28), cmap='gray')
axes[0].set_title('Original Image')
axes[1].imshow(np.real(reconstructed_image.reshape(28, 28)), cmap='gray')
axes[1].set_title(f'Reconstructed Image using {n_pcs_90} PCs')
plt.show()