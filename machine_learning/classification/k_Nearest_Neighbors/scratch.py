# %%
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import operator

split = 0.66
n_neighbors = 15
h = .02  # step size in the mesh

# %%
# import some data to play with
iris = datasets.load_iris()
iris

X = iris.data[:, :2]
y = iris.target

# preprocess dataset, split into training and test part
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=.4, random_state=42)

# %%
Xy_train = np.append(X_train, y_train.reshape(-1, 1), 1)
X_train_class_0 = np.asarray(Xy_train[Xy_train[:, 2] == 0])[:, :2]
X_train_class_1 = np.asarray(Xy_train[Xy_train[:, 2] == 1])[:, :2]
X_train_class_2 = np.asarray(Xy_train[Xy_train[:, 2] == 2])[:, :2]
Xy_test = np.append(X_test, y_test.reshape(-1, 1), 1)
X_test_class_0 = np.asarray(Xy_test[Xy_test[:, 2] == 0])[:, :2]
X_test_class_1 = np.asarray(Xy_test[Xy_test[:, 2] == 1])[:, :2]
X_test_class_2 = np.asarray(Xy_test[Xy_test[:, 2] == 2])[:, :2]

# %%
# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# %%
np.c_[xx.ravel(), yy.ravel()]

# %%
# square root of the sum of the squared differences between the two arrays of numbers


def euclidean_distance(vector1, vector2):
    return np.sqrt(np.sum(np.power(vector1-vector2, 2)))


def absolute_distance(vector1, vector2):
    return np.sum(np.absolute(vector1-vector2))


def get_neighbours(X_train, X_test_instance, k):
    distances = []
    neighbors = []
    for i in range(0, X_train.shape[0]):
        dist = absolute_distance(X_train[i], X_test_instance)
        distances.append((i, dist))
    distances.sort(key=operator.itemgetter(1))
    for x in range(k):
        # print distances[x]
        neighbors.append(distances[x][0])
    return neighbors


def predictkNNClass(output, y_train):
    classVotes = {}
    for i in range(len(output)):
        #         print output[i], y_train[output[i]]
        if y_train[output[i]] in classVotes:
            classVotes[y_train[output[i]]] += 1
        else:
            classVotes[y_train[output[i]]] = 1
    sortedVotes = sorted(classVotes.items(),
                         key=operator.itemgetter(1), reverse=True)
    # print sortedVotes
    return sortedVotes[0][0]


def kNN_test(X_train, X_test, Y_train, Y_test, k):
    output_classes = []
    for i in range(0, X_test.shape[0]):
        output = get_neighbours(X_train, X_test[i], k)
        predictedClass = predictkNNClass(output, Y_train)
        output_classes.append(predictedClass)
    return output_classes


def prediction_accuracy(predicted_labels, original_labels):
    count = 0
    for i in range(len(predicted_labels)):
        if predicted_labels[i] == original_labels[i]:
            count += 1
    # print count, len(predicted_labels)
    return float(count)/len(predicted_labels)


def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0


# %%
predicted_classes = {}
final_accuracies = {}
# for k in range(1, 21):
predicted_classes = np.array(
    kNN_test(X_train, X_test, y_train, y_test, n_neighbors))
final_accuracies = prediction_accuracy(predicted_classes, y_test)

# %%
predicted_classes

# %%
final_accuracies

# %%
Z_test = predicted_classes

# %%
# Put the result into a color plot
Z = Z_test.reshape(-1, 1).flat()
Z
