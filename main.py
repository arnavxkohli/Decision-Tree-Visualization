import src
import time
import matplotlib.pyplot as plt
import os
import shutil
import numpy as np


def main():
    """
    Main function.
    """
    start = time.time()

    input_args = src.parser.arg_parser()

    if input_args.training_dataset:
        # load labels and attributes from file
        y, x = src.parser.file_parser(file = input_args.training_dataset)
        path = input_args.training_dataset.split('/')

        directory = ''
        for i, dir in enumerate(path):
            if i != len(path) - 1:
                directory += dir

        # initialize to None for ease of error handling
        image_directory = None
        file_name = None

        # handle directory creation for the images
        try:
            image_directory = f'img/{directory}'
            file_name = path[-1].split('.txt')[0]

            if input_args.regenerate_images:
                if os.path.exists(image_directory):
                    shutil.rmtree(image_directory)

            os.makedirs(image_directory, exist_ok=True)
        except NameError as e:
            print(f'Error extracting paths: {e}')
        except IndexError as e:
            print(f'Error extracting paths: {e}')
        except AttributeError as e:
            print(f'Error extracting paths: {e}')
        except TypeError as e:
            print(f'Error extracting paths: {e}')
        except Exception as e:
            print(f'Error extracting paths: {e}')

        print(f'\nDataset: {input_args.training_dataset}\n')

        if input_args.visualize:
            print('Entering visualization of entire dataset...\n')
            # generate decision tree from all provided data

            if input_args.prune:
                split_ratio = 0.2  # 80:20 split
                num_samples = len(x)
                n_val = int(split_ratio * num_samples)
                indices = np.random.permutation(num_samples)

                val_indices, train_indices = indices[:n_val], indices[n_val:]

                # Create training and validation sets
                val_x, new_x = x[val_indices], \
                               x[train_indices]
                val_y, new_y = y[val_indices], \
                               y[train_indices]


                tree, depth = \
                    src.decision_tree.decision_tree_learning(y = new_y, \
                                                             x = new_x)

                tree, depth = \
                    src.decision_tree.decision_tree_pruning(tree = tree, \
                                                            train_x = new_x, \
                                                            train_y = new_y, \
                                                            val_y = val_y, \
                                                            val_x = val_x)
            else:
                tree, depth = \
                    src.decision_tree.decision_tree_learning(y = y, x = x)

            print(f'Depth of trained decision tree: {depth}\n')

            plt.figure(figsize=(20, 20))
            plt.axis('off')
            # plot the entire tree
            tree.plot_tree()
            # only save if directory and file were parsed properly
            if image_directory and file_name:
                try:
                    plt.savefig(f'{image_directory}/{file_name}_tree.png', \
                                format = 'png')
                except FileNotFoundError as e:
                    print(f'Error saving tree to folder: {e}')
                except ValueError as e:
                    print(f'Error saving tree to folder: {e}')
                except PermissionError as e:
                    print(f'Error saving tree to folder: {e}')
                except TypeError as e:
                    print(f'Error saving tree to folder: {e}')
                except Exception as e:
                    print(f'Error saving tree to folder: {e}')
            plt.show()

            if not input_args.prune:
                # Only perform sanity check on unpruned tree
                for i in range(y.size):
                    if tree.predict(x[i]) != y[i]:
                        # generated tree should be able to perfectly predict
                        # training data re-entered as test data, initial
                        # sanity check
                        error_text = \
                            f'Wrong prediction for {x[i]}: expected \'{y[i]}\''
                        print(f'Info: {error_text}')

            print()
        # perform k fold cross validation, default k is 10, return all the
        # calculated values
        average_precisions, average_recalls, \
        average_confusion_matrix, average_accuracy, f1 = \
            src.evaluate.k_fold_cross_validation(y = y, x = x, \
                                                 k = input_args.folds, \
                                                 seed = input_args.seed, \
                                                 prune = input_args.prune)

        print('\nObtained Values:\n')
        print('Precisions', average_precisions)
        print('Recalls', average_recalls)
        print('Accuracy', average_accuracy)
        print('F1', f1)
        # plot confusion matrix heat map
        print('\nEntering confusion matrix visualization...\n')
        src.evaluate.plot_confusion_matrix(confusion_matrix = \
                                        average_confusion_matrix)
        # only save if directory and file were parsed properly
        if image_directory and file_name:
            extension = '_confusion_matrix_heat_map.png'
            try:
                plt.savefig(f'{image_directory}/{file_name}{extension}', \
                            format='png')
            except FileNotFoundError as e:
                print(f'Error saving heat map to folder: {e}')
            except ValueError as e:
                print(f'Error saving heat map to folder: {e}')
            except PermissionError as e:
                print(f'Error saving heat map to folder: {e}')
            except TypeError as e:
                print(f'Error saving heat map to folder: {e}')
            except Exception as e:
                print(f'Error saving heat map to folder: {e}')
        plt.show()

    print(f'Script took: {int(time.time() - start)} seconds to run.\n')


if __name__ == "__main__":
    main()
