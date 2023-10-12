# Author: 
# Date:
# Project: 
# Acknowledgements: 
#


from typing import Union
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

from tools import load_iris, split_train_test


features, targets, classes = load_iris()
[n, f_dim] = features.shape


def prior(targets: np.ndarray, classes: list) -> np.ndarray:
    '''
    Calculate the prior probability of each class type
    given a list of all targets and all class types
    '''
    
    prob_list = np.zeros(len(classes))
    n = len(targets)
    for index, c in enumerate(classes):
        cnt = np.sum(targets == c)
        prob_list[index] = cnt / n
    return prob_list
    return prob_list
# print(prior([0,2,3,3],[0,1,2,3]))


def split_data(
    features,
    targets: np.ndarray,
    split_feature_index: int,
    theta: float
) -> Union[tuple, tuple]:
    '''
    Split a dataset and targets into two seperate datasets
    where data with split_feature < theta goes to 1 otherwise 2
    '''
    mask_1 = features[:, split_feature_index] < theta
    mask_2 = ~mask_1  # Inverse of mask_1
    # Apply the masks to split the data and targets
    features_1 = features[mask_1]
    targets_1 = targets[mask_1]
    
    features_2 = features[mask_2]
    targets_2 = targets[mask_2]

    return (features_1, targets_1), (features_2, targets_2)
(f_1, t_1), (f_2, t_2) = split_data(features, targets, 2, 4.65)
# print(len(f_1), len(f_2))

def gini_impurity(targets: np.ndarray, classes: list) -> float:
    '''
    Calculate:
        i(S_k) = 1/2 * (1 - sum_i P{C_i}**2)
    '''
    return (1-np.sum(prior(targets,classes)**2))/2
# print(gini_impurity(t_2,classes))


def weighted_impurity(
    t1: np.ndarray,
    t2: np.ndarray,
    classes: list
) -> float:
    '''
    Given targets of two branches, return the weighted
    sum of gini branch impurities
    '''
    g1 = gini_impurity(t1, classes)
    g2 = gini_impurity(t2,classes)
    n = t1.shape[0] + t2.shape[0]
    return t1.shape[0]*g1/n +t2.shape[0]*g2/n
# print(weighted_impurity(t_1,t_2,classes))


def total_gini_impurity(
    features: np.ndarray,
    targets: np.ndarray,
    classes: list,
    split_feature_index: int,
    theta: float
) -> float:
    (f_1, t_1), (f_2, t_2) = split_data(features, targets, split_feature_index, theta)
    return weighted_impurity(t_1,t_2,classes)
# total_gini_impurity(features, targets, classes, 2, 4.65)


def brute_best_split(
    features: np.ndarray,
    targets: np.ndarray,
    classes: list,
    num_tries: int
) -> Union[float, int, float]:
    '''
    Find the best split for the given data. Test splitting
    on each feature dimension num_tries times.

    Return the lowest gini impurity, the feature dimension and
    the threshold
    '''
    best_gini, best_dim, best_theta = float("inf"), None, None
    # iterate feature dimensions
    for i in range(features.shape[1]):
        # create the thresholds
        min_val, max_val = np.min(features[:,i]), np.max(features[:,i])
        thetas = np.linspace(min_val, max_val, num_tries)
        # iterate thresholds

        for theta in thetas:
            gini = total_gini_impurity(features,targets,classes,i,theta)
            if gini < best_gini:
                best_gini = gini
                best_dim = i
                best_theta = theta
    return best_gini, best_dim, best_theta
# print(brute_best_split(features, targets, classes, 30))


class IrisTreeTrainer:
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        classes: list = [0, 1, 2],
        train_ratio: float = 0.8
    ):
        '''
        train_ratio: The ratio of the Iris dataset that will
        be dedicated to training.
        '''
        (self.train_features, self.train_targets),\
            (self.test_features, self.test_targets) =\
            split_train_test(features, targets, train_ratio)

        self.classes = classes
        self.tree = DecisionTreeClassifier()

    def train(self):
        self.tree.fit(self.train_features, self.train_targets)

    def accuracy(self):
        # Make predictions on the test data
        predictions = self.tree.predict(self.test_features)
        # Calculate accuracy by comparing predicted labels to true labels
        acc = accuracy_score(self.test_targets, predictions)
        return acc
    def plot(self):
        plt.figure(figsize=(12, 6))  # Adjust the figure size as needed
        plot_tree(self.tree, filled=True)
        plt.savefig('2_3_1.png')
        plt.show()
    def guess(self):
        # Make predictions on the test data features
        predictions = self.tree.predict(self.test_features)
        return predictions
    def confusion_matrix(self):
        # Make predictions on the test data features
        predictions = self.tree.predict(self.test_features)
        
        # Initialize the confusion matrix
        num_classes = len(self.classes)
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
        
        # Populate the confusion matrix
        for true_label, predicted_label in zip(self.test_targets, predictions):
            confusion_matrix[true_label][predicted_label] += 1
        
        return confusion_matrix
features, targets, classes = load_iris()
dt = IrisTreeTrainer(features, targets, classes=classes)
dt.train()
print(f'The accuracy is: {dt.accuracy()}')
dt.plot()
print(f'I guessed: {dt.guess()}')
print(f'The true targets are: {dt.test_targets}')
print(dt.confusion_matrix())

