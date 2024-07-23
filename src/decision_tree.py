import numpy as np
import matplotlib.pyplot as plt
from . import entropy


class DecisionTree:
    """
    Decision tree object.
    """
    def __init__(self, root: int | tuple[int, float], \
                 left: int | tuple[int, float] | None = None, \
                 right: int | tuple[int, float] | None = None):
        """
        Constructor

        Args:
            root (int | tuple[int, float]): Root node
            left (int | tuple[int, float] | None): (Optional) Node to the
            left of the root node, captures values that are below the
            splitting value for the dataset.
            right (int | tuple[int, float] | None): (Optional) Node to the
            right of the root node, captures values that are above the
            splitting value for the dataset.

            If both left and right are None, then the node is a leaf node.

            All three arguments take a union of int and tuple types. At a
            leaf node the value is an int, and at every other node, the
            value is a tuple as explained above.
        """
        self.root = root
        self.left = left
        self.right = right

    def predict(self, x: np.ndarray) -> int:
        """
        Prediction function, given an attribute array as input returns the
        predicted value according to the decision tree already built.
        Iteratively goes through each decision node until it reaches a leaf
        node (which has to be a pure node with the appropriate label for the
        prediction to hold).

        Args:
            x (np.ndarray): Input attribute numpy array with shape (, K),
            taken from the test data. K is the number of attributes in the
            array.

        Returns:
            int: Prediction for the given input.
        """
        while self.left and self.right:
            # recursively seek left if attribute less than split
            if x[self.root[0]] < self.root[1]:
                self = self.left
            # recursively seek right if attribute greater than split
            else:
                self = self.right

        return self.root

    def plot_tree(self, x=0, y=0, depth: int = 0):
        """
        Print out the class label if leaf nodes, else print the split
        condition. Recursively visualise left and right subsets.

        Args:
            depth (int): The current depth in the tree.
        """
        box_width = 600
        box_height = 20

        if not self.left and not self.right:
            leaf_text = f'{self.root}'
        else:
            attribute, split_value = self.root
            split_text = f'x{attribute} < {split_value}'

        node_text = leaf_text if (not self.left and not self.right) \
                    else split_text
        plt.text(x, y, node_text, ha = 'center', va = 'center', \
                 bbox = dict(facecolor='lightblue', alpha=0.7, \
                             boxstyle='round'), fontsize=7)

        # Calculate the dynamic horizontal spacing
        horizontal_spacing = box_width / 2 ** depth if depth > 1 \
                             else box_width - depth * 250

        if self.left:
            x_next = x - horizontal_spacing
            plt.plot([x, x_next], [y - box_height, y - 5 * box_height], 'k-')
            self.left.plot_tree(x_next, y - 5 * box_height, depth+1)

        if self.right:
            x_next = x + horizontal_spacing
            plt.plot([x, x_next], [y - box_height, y - 5 * box_height], 'k-')
            self.right.plot_tree(x_next, y - 5 * box_height, depth+1)


def decision_tree_learning(y: np.ndarray, x: np.ndarray, depth: int = 0) -> \
    tuple[DecisionTree, int]:
    """
    Recursive function that builds the decision tree for given labels and
    attribute arrays. Returns the depth of the tree and a pointer to the root
    node of the tree.

    Args:
        y (np.ndarray): Numpy array of shape (N, ) where N is the number of
        instances. Contains the labels corresponding to each element in the
        set of attributes.
        x (np.ndarray): Numpy array of shape (N, K) where N is the number of
        instances and K is the number of attributes. Contains the input
        elements corresponding to each attribute.
        depth (int): The depth of the tree. Mainly used for the recursion.
        Default is set to 0.

    Returns:
        tuple[DecisionTree, int]: Tuple containing a pointer to the root of
        the tree and the depth of the tree after it has been built.
    """
    # leaf node only has one label
    if np.unique(y).size == 1:
        # if leaf then return node with the value as the label
        return DecisionTree(root = np.unique(y)[0]), depth
    # if more than one label, need to split
    else:
        # find split condition
        split_attribute, split_value = entropy.InformationGain(y = y, x = x) \
                                       .find_split()
        # apply filter using split condition
        left_y = y[x[:, split_attribute] < split_value]
        right_y = y[x[:, split_attribute] >= split_value]
        left_x = x[x[:, split_attribute] < split_value]
        right_x = x[x[:, split_attribute] >= split_value]
        # create a node with a tuple of the split conditions as the value
        node = DecisionTree(root = (split_attribute, split_value))
        # left recursion based on split condition
        node.left, left_depth = decision_tree_learning(y = left_y, \
                                                       x = left_x, \
                                                       depth = depth + 1)
        # right recursion based on split condition
        node.right, right_depth = decision_tree_learning(y = right_y, \
                                                         x = right_x, \
                                                         depth = depth + 1)
        # finally, return pointer to root node and the true depth of tree
        return node, max(left_depth, right_depth)


def decision_tree_pruning(tree:DecisionTree, train_x: np.ndarray, \
                          train_y: np.ndarray, val_y: np.ndarray, \
                          val_x: np.ndarray, depth: int = 0) -> \
                          tuple[DecisionTree, int]:
    """
    Recursive function that prunes the decision tree for given labels and
    attribute arrays which is used as validation set. Returns the depth of the
    tree and a pointer to the root node of the tree.

    Args:
        tree (DecisionTree): DecisionTree object that is a pointer to the
        trained decision tree.
        val_y (np.ndarray): Numpy array of shape (N, ) where N is the number
        of instances. Contains the labels corresponding to each element in the
        set of attributes.
        val_x (np.ndarray): Numpy array of shape (N, K) where N is the number
        of instances and K is the number of attributes. Contains the input
        elements corresponding to each attribute.
        depth (int): The depth of the tree. Mainly used for the recursion.
        Default is set to 0.

    Returns:
        tuple[DecisionTree, int]: Tuple containing a pointer to the root of
        the tree and the depth of the tree after it has been pruned.
    """
    if tree.left == None:
        return tree, depth

    val_indices = val_x[:,tree.root[0]] < tree.root[1]
    train_indices = train_x[:,tree.root[0]] < tree.root[1]

    left_root, left_depth = decision_tree_pruning(tree.left, \
                        train_x[train_indices], train_y[train_indices], \
                        val_y[val_indices], val_x[val_indices], depth+1)

    right_root, right_depth = decision_tree_pruning(tree.right, \
                        train_x[~train_indices], train_y[~train_indices], \
                        val_y[~val_indices], val_x[~val_indices], depth+1)

    pre_accuracy = sum([tree.predict(attribute)==label \
                        for attribute, label in zip(val_x, val_y)])

    unique_values, counts = np.unique(train_y[train_indices], \
                                      return_counts=True)
    max_count_index = np.argmax(counts)
    tree.left = DecisionTree(root = unique_values[max_count_index])

    unique_values, counts = np.unique(train_y[~train_indices], \
                                      return_counts=True)
    max_count_index = np.argmax(counts)
    tree.right = DecisionTree(root = unique_values[max_count_index])

    post_accuracy = sum([tree.predict(attribute)==label \
                        for attribute, label in zip(val_x, val_y)])

    if post_accuracy < pre_accuracy:
        tree.left = left_root
        tree.right = right_root
        depth = max(left_depth,right_depth)

    else:
        depth = depth + 1

    return tree, depth
