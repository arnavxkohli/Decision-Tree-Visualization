import numpy as np
import argparse


def file_parser(file: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Read in the dataset from the specified filepath.

    Args:
        filename (str): The filepath to the dataset file.

    Returns:
        tuple[np.ndarray, np.ndarray]: Returns a tuple of (x, y), each being
        a numpy array. x has shape (N, K) and y has shape (N, )
        where N is the number of instances and K is the number of
        features or attributes.
    """
    data = np.loadtxt(file)
    columns = len(data[0])
    # split data into labels and attributes based on the columns
    y, x = data[:, columns-1], data[:, :columns-1]

    return y, x


def arg_parser() -> argparse.Namespace:
    """
    Handles parsing of arguments from the command line. Determines if
    visualization, cross validation or both are run while testing the program.
    Helpful in making the code interactive an easy to test, providing
    customizations for the k value in the cross validation and the seed
    value used for the random generator while splitting datasets. Also allows
    pruning to take place.

    Returns:
        argparse.Namespace: Namespace containing the required command line
        arguments to run the tests and provide functionality to the
        application.
            - training_dataset: File path for the training dataset.
            - visualize: Visualize the decision tree built on the entire
            training data.
            - folds: 'k' value determining the folds for the cross validation
            test.
            - seed: Seed for the cross validation random generator.
            - prune: Whether to prune the decision tree or not.
            - regenerate_images: Whether to completely regenerate the img/
            folder.
    """
    parser = argparse.ArgumentParser(description = 'Decision Tree CLI')

    parser.add_argument('training_dataset', help = 'File path for the \
                        training dataset',
                        default= 'wifi_db/clean_dataset.txt')

    parser.add_argument('--visualize', help = 'Visualize the decision tree \
                        built on the entire training data',
                        action='store_true')

    parser.add_argument('--folds', help = '\'k\' value determining the folds \
                        for the cross validation test', type = int,
                        default = 10)

    parser.add_argument('--seed', help = 'Seed for the cross validation \
                        random generator', type = int, default = 63000)

    parser.add_argument('--prune', help = 'Prune the decision tree', \
                        action = 'store_true')

    parser.add_argument('--regenerate_images', help = 'Delete the images \
                        folder and regenerate all of the images within it', \
                        action = 'store_true')

    return parser.parse_args()
