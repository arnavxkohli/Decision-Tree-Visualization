import numpy as np
import matplotlib.pyplot as plt
from . import decision_tree
from numpy.random import default_rng


def evaluate(trained_tree: decision_tree.DecisionTree, \
             test_y: np.ndarray, test_x: np.ndarray) -> float:
    """
    Given an already trained decision tree, takes as input test attributes
    which are evaluated with the predict method of the tree, and compared with
    the test labels outputing the accuracy of the tree.

    Args:
        trained_tree (DecisionTree): The decision tree already trained on the
        training data.
        test_y (np.ndarray): Labels corresponding to the testing dataset.
        test_x (np.ndarray): Attribute sets corresponding to the testing
        dataset.

    Returns:
        float: The returned accuracy.
    """
    correctly_classified = 0
    total = len(test_x)

    for i in range(total):
        prediction = trained_tree.predict(test_x[i])
        if prediction == test_y[i]:
            correctly_classified += 1

    accuracy = correctly_classified / total
    return accuracy


def generate_confusion_matrix(trained_tree: decision_tree.DecisionTree, \
             test_y: np.ndarray, test_x: np.ndarray) -> float:
    """
    Given an already trained decision tree, takes as input test attributes
    which are evaluated with the predict method of the tree, and compared with
    the test labels outputing the accuracy of the tree.
    the test labels outputing the confusion matrix of the tree.

    Args:
        trained_tree (DecisionTree): The decision tree already trained on the
        dataset.

    Returns:
        float: The returned accuracy.
        np.ndarray: Confusion Matrix. It is 4x4 since there are 4 unique
        classes in the datasets.
    """

    num_labels = len(np.unique(test_y))
    confusion_matrix = np.zeros(shape=(num_labels,num_labels))

    for attributes, label in zip(test_x, test_y):
        predicted = trained_tree.predict(attributes)
        confusion_matrix[int(label-1)][int(predicted-1)]+=1

    return confusion_matrix


def plot_confusion_matrix(confusion_matrix: np.ndarray):
        """
        Plot heat map for the confusion matrix to better visualisation
        and understanding of the accuracy of the model.

        Args:
            confusion_matrix (np.ndarray): The numpy array which
            represents the confusion matrix.
        """

        plt.clf()
        plt.axis('on')
        plt.imshow(confusion_matrix, interpolation='nearest', \
                   cmap=plt.cm.cool)
        plt.title('Confusion Matrix Heatmap')
        plt.colorbar()

        classes = ['Room {}'.format(i+1) for i in \
                   range(confusion_matrix.shape[0])]

        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                plt.text(j, i, str(confusion_matrix[i, j]), \
                         horizontalalignment='center', color='black')

        plt.xlabel('Predicted')
        plt.ylabel('Actual')


def k_fold_cross_validation(y: np.ndarray, x: np.ndarray, \
                            k: int = 10, seed: int = 63000, \
                            prune: bool = False) -> np.ndarray:
    """
    Performs a k cross validation on the dataset provided, splitting it
    accordingly. Prints accuracy per class, overall accuracy and confusion
    matrix into the terminal. Also prints precision, recall and f1 measure
    per class.

    Args:
        y (np.ndarray): Whole input labels dataset. Split according to the
        order of the validation (k).
        x (np.ndarray): Whole input attributes dataset. Split according to
        the order of the validation (k).
        k (int): Order of the cross validation. Decides how the data will be
        split up.
        seed (int): Seeding value for the cross validation.
        prune (bool): If pruning should be performed
    Returns:
        np.ndarray: Confusion Matrix. It is 4x4 since there are 4 unique
        classes in the datasets. Also returns recall, precision and f1
        measures.
    """
    random_generator=default_rng(seed = seed)
    n_instances = len(x)
    shuffled_indices = random_generator.permutation(n_instances)


    accuracy_sum = 0
    confusion_matrix_sum = 0
    precisions = np.zeros(len(np.unique(y)))
    recalls = np.zeros(len(np.unique(y)))


    for fold in range(k):
        print('Running fold: ', fold + 1)
        split_indices = np.array(np.array_split(shuffled_indices, k))
        test_indices = split_indices[fold]

        train_indices = np.concatenate([split_indices[:fold].flatten(), \
                                        split_indices[fold+1:].flatten()])

        train_x, train_y = x[train_indices], y[train_indices]
        test_x, test_y = x[test_indices], y[test_indices]

        if prune:
            split_ratio = 0.2  # 80:20 split
            num_samples = len(train_x)
            num_validation_samples = int(split_ratio * num_samples)

            # Create training and validation sets
            val_x, train_x = train_x[:num_validation_samples], \
                            train_x[num_validation_samples:]
            val_y, train_y = train_y[:num_validation_samples], \
                            train_y[num_validation_samples:]

            trained_tree_output = \
                decision_tree.decision_tree_learning(train_y, train_x)
            pruned_tree = \
                decision_tree.decision_tree_pruning(trained_tree_output[0], \
                                                    train_x, train_y, \
                                                    val_y, val_x)[0]
            final_tree = pruned_tree

        else:
            trained_tree_output = \
                decision_tree.decision_tree_learning(train_y, train_x)
            final_tree = trained_tree_output[0]

        accuracy_sum += evaluate(final_tree, test_y, test_x)
        confusion_matrix = \
            generate_confusion_matrix(final_tree, test_y, test_x)
        confusion_matrix_sum += confusion_matrix

        for i in range(confusion_matrix.shape[0]):
            column_sum = np.sum(confusion_matrix[:, i])
            row_sum = np.sum(confusion_matrix[i,:])
            precision = confusion_matrix[i,i]/column_sum
            recall = confusion_matrix[i,i]/row_sum
            precisions[i] += precision
            recalls[i] += recall

    average_precisions = precisions / k
    average_recalls = recalls / k
    average_confusion_matrix = confusion_matrix_sum/k
    average_accuracy = accuracy_sum/k
    f1 = (2*average_precisions*average_recalls) / \
         (average_precisions + average_recalls)

    return (np.round(average_precisions, 2), np.round(average_recalls, 2), \
            np.round(average_confusion_matrix, 2), \
            np.round(average_accuracy, 2), np.round(f1, 2))
