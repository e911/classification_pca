import numpy as np
from matplotlib import pyplot as plt


def load_data(file_name):
    return np.loadtxt(file_name)


train_data = np.loadtxt('dataCodeClass/traindata3C.txt')
test_data = np.loadtxt('dataCodeClass/testdata3C.txt')

train_labels = np.loadtxt('dataCodeClass/trainlabel3C.txt')
test_labels = np.loadtxt('dataCodeClass/testlabel3C.txt')

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


Z_train = x_data_preprocess(train_data, 2)
Z_test = x_data_preprocess(test_data,2)

nclass = len(np.unique(train_labels))
ntrain = len(train_labels)

Y_train = np.zeros((nclass, ntrain))

for i in range(ntrain):
    Y_train[int(train_labels[i]) - 1, i] = 1
W = np.dot(Y_train, np.dot(Z_train.T, np.linalg.inv(np.dot(Z_train, Z_train.T)) ))

Y_pred = np.dot(W, Z_test)

predicted_labels = np.argmax(Y_pred, axis=0) + 1
accuracy = np.sum(predicted_labels == test_labels) / len(test_labels) * 100
print(f'Accuracy: {accuracy:.2f}%')
print('Weight Matrix W:')
print(W)


fig, ax = plt.subplots(figsize=(10, 8))

ax.scatter(train_data[:, 0], train_data[:, 1], c=train_labels, marker='o', label='Training Data')
ax.scatter(test_data[:, 0], test_data[:, 1], c=predicted_labels, marker='x', label='Test Data (Predicted)')

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_title('Scatter Matrix with Training and Test Data')
ax.legend()

plt.show()