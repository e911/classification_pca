# A function to calculate the pca  of a vector X

import numpy as np


def pca(x, type):
    mean_x = np.mean(x, axis=0)
    center_x = x - mean_x

    x_xt = np.cov(center_x, rowvar=False)

    if type == 'eigen':
        eigenvalues, eigenvectors = np.linalg.eig(x_xt)
        sorted_eigens = np.argsort(eigenvalues)[::-1]
        sorted_eigenvectors = eigenvectors[:, sorted_eigens]
        k = 2
        top_k_eigenvectors = sorted_eigenvectors[:, :k]
        center_x = center_x[:, :k]
        new_bias_data = center_x.dot(top_k_eigenvectors.T)

        print(x)
        print(np.add(new_bias_data, np.expand_dims(mean_x, axis=1)))

    elif type == 'svd':
        _, _, vh = np.linalg.svd(x, full_matrices=False)
        k = 2
        top_k_singular_vectors = vh[:k, :].T
        new_data = x.dot(top_k_singular_vectors)
        print(x)
        print(new_data)

x=np.array([[1,2,3],[4,5,6],[7,8,9]])
pca(x,'eigen')
pca(x,'svd')


