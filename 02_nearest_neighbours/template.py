# Author: Arnar Gylfi Haraldsson
# Date: 3.09.2023
# Project: 
# Acknowledgements: 
#

import numpy as np
import matplotlib.pyplot as plt

from tools import load_iris, split_train_test, plot_points

d, t, classes = load_iris()
x, points = d[0,:], d[1:, :]
x_target, point_targets = t[0], t[1:]

def euclidian_distance(x: np.ndarray, y: np.ndarray) -> float:
    '''
    Calculate the euclidian distance between points x and y
    '''
    return np.sqrt(np.sum((x - y) ** 2))


def euclidian_distances(x: np.ndarray, points: np.ndarray) -> np.ndarray:
    '''
    Calculate the Euclidean distances between x and a set of points
    '''
    distances = np.zeros(points.shape[0])
    for i in range(points.shape[0]):
        distances[i] = euclidian_distance(x, points[i])
    return distances

def k_nearest(x: np.ndarray, points: np.ndarray, k: int):
    distances = euclidian_distances(x, points)
    sorted_indices = np.argsort(distances)    
    # Return the first k indices (k-nearest points)
    return sorted_indices[:k]


def vote(targets, classes):
    '''
    Given a list of nearest targets, vote for the most
    popular
    '''
    class_counts = np.bincount(targets)
    most_common_class = classes[np.argmax(class_counts)]    
    return most_common_class


def knn(
    x: np.ndarray,
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    '''
    Combine k_nearest and vote
    '''
    nearest_indices = k_nearest(x, points, k)
    nearest_targets = point_targets[nearest_indices]
    predicted_class = vote(nearest_targets, classes)
    return predicted_class

def knn_predict(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    num_points = points.shape[0]
    predictions = np.zeros(num_points, dtype=int)
    for i in range(num_points):
        x = points[i, :]
        predictions[i] = knn(x, points, point_targets, classes, k)
    return predictions
(d_train, t_train), (d_test, t_test) = split_train_test(d, t, train_ratio=0.8)


def knn_accuracy(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> float:
    predictions = knn_predict(points, point_targets, classes, k)
    accuracy = np.mean(predictions == point_targets)
    return accuracy

def knn_confusion_matrix(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    num_classes = len(classes)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    predictions = knn_predict(points, point_targets, classes, k)
    for true_label, predicted_label in zip(point_targets, predictions):
        confusion_matrix[true_label][predicted_label] += 1
    return confusion_matrix


def best_k(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
) -> int:
    num_points = points.shape[0]
    best_accuracy = 0.0
    best_k_value = 0
    for k in range(2, num_points-1): #Always returns one if iteration starts at k = 1
        accuracy = knn_accuracy(points, point_targets, classes, k)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k_value = k
    return best_k_value



def knn_plot_points(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
):
    predictions = knn_predict(points, point_targets, classes, k)
    
    # Define colors for correct and incorrect predictions
    correct_color = 'green'
    incorrect_color = 'red'
    
    for i, point in enumerate(points):
        plt.scatter(point[0], point[1], c='black', marker='o', label=f'Class {int(point_targets[i])}')
        
        if point_targets[i] == predictions[i]:
            edge_color = correct_color
        else:
            edge_color = incorrect_color
        
        plt.gca().add_patch(plt.Circle((point[0], point[1]), 0.1, color=edge_color, fill=False, lw=2))
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'(k={k})')
    plt.grid(True)
    plt.axis('equal')
    plt.show()
knn_plot_points(d, t, classes, 3)


def weighted_vote(
    targets: np.ndarray,
    distances: np.ndarray,
    classes: list
) -> int:
    '''
    Given a list of nearest targets, vote for the most
    popular
    '''
    # Remove if you don't go for independent section
    ...


def wknn(
    x: np.ndarray,
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    '''
    Combine k_nearest and vote
    '''
    # Remove if you don't go for independent section
    ...


def wknn_predict(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    # Remove if you don't go for independent section
    ...


def compare_knns(
    points: np.ndarray,
    targets: np.ndarray,
    classes: list
):
    # Remove if you don't go for independent section
    ...
