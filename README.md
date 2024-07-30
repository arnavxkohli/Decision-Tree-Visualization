# Decision Trees

## Table of contents

- [Table of contents](#table-of-contents)
- [Running Instructions](#running-instructions)
  - [Overview](#overview)
  - [Setting up the virtual environment](#setting-up-the-virtual-environment)
  - [General Usage](#general-usage)
  - [Specific Example Usage](#specific-example-usage)
- [Directory Structure](#directory-structure)

## Running Instructions

### Overview

The code provided supports three main functionalities:

- Inputing a file as a command line argument, this can be used as the testing dataset (for ease of use with the secret dataset).
- Visualization of the decision tree generated.
- `k`-fold (default set to `10`) cross validation and calculation of the corresponding parameters (precision, accuracy, f1 measures, confusion matrix).
- Pruning of decision tree generated.

In addition to this, the code also supports:

- Changing the value of `k`.
- Changing the seed, relevant for the cross validation step.
- Regenerating the images folder `img/` to verify the images generated are valid.

While visualizing the tree, `png` format images are produced.

### Setting up the virtual environment

To run this code on a lab machine, setting up the same virtual environment provided with the initial code is recommended. Development was undertaken within this environment and it can be configured by running:

```bash
python3 -m venv venv
source venv/bin/activate
```

And within the virtual environment (indicated by a venv next to the username on the terminal), running:

``` bash
pip install --upgrade pip
pip install -r requirements.txt
```

The virtual environment is optional (but recommended) however, and the script will run on any lab machine with all the dependencies present in `requirements.txt` already installed without the need of a virtual environment.

### General Usage

Still within the virtual environment, the script can be run using:

```bash
python3 main.py <path/to/input>
```

Where `<path/to/input>` can be replaced with the secret test dataset. Images generated for this dataset will be automatically added if ran with:

```bash
python3 main.py <path/to/input> --visualize
```

This is optional however, and method 1 is recommended.

Usage info for the command line interface program is given below:

```text
usage: main.py [-h] [--visualize] [--folds FOLDS] [--seed SEED] [--prune] [--regenerate_images] training_dataset

Decision Tree CLI

positional arguments:
  training_dataset     File path for the training dataset

options:
  -h, --help           show this help message and exit
  --visualize          Visualize the decision tree built on the entire training data
  --folds FOLDS        'k' value determining the folds for the cross validation test
  --seed SEED          Seed for the cross validation random generator
  --prune              Prune the decision trees.
  --regenerate_images  Delete the images folder and regenerate all of the images within it
```

### Specific Example Usage

As an example, `wifi_db/clean_dataset.txt` can be used as an input by running:

```bash
python3 main.py wifi_db/clean_dataset.txt --visualize
```

Similarly, for `wifi_db/noisy_dataset.txt`:

```bash
python3 main.py wifi_db/noisy_dataset.txt --visualize
```

Pruning can be enabled by adding the `--prune` flag.
Visualization and pruning can be disabled. As an example (using `wifi_db/clean_dataset.txt`):

```bash
python3 main.py wifi_db/clean_dataset.txt --visualize
```

This only visualizes and cross validates the generated tree.

```bash
python3 main.py wifi_db/clean_dataset.txt
```

This only cross validates the generated tree.

```bash
python3 main.py wifi_db/clean_dataset.txt --visualize --prune
```

This prunes, cross validates and visualizes the decision tree and should be the default run setting.

```bash
python3 main.py wifi_db/clean_dataset.txt --visualize --prune --regenerate_images
```

This performs visualization, pruning and also regenerates the images in `img/`.

## Directory Structure

The project was divided according to the following structure:

```text
decision-trees-coursework
├── Clean Dataset Visualization.mp4
├── README.md
├── img
│   └── wifi_db
├── main.py
├── report.pdf
├── requirements.txt
├── spec.pdf
├── src
│   ├── __init__.py
│   ├── __pycache__
│   ├── decision_tree.py
│   ├── entropy.py
│   ├── evaluate.py
│   └── parser.py
├── venv
│   ├── bin
│   ├── include
│   ├── lib
│   ├── pyvenv.cfg
│   └── share
└── wifi_db
    ├── clean_dataset.txt
    └── noisy_dataset.txt
```

- The `src/` folder contains all of the python files with the necessary functions to build the decision tree, evaluate it and parse file and command line inputs. This is standard naming convention for a module, and an `__init__.py` file was also included to ensure the python interpreter recognized `src/` as a module.
- `venv` is the virtual environment which is set-up following the dependencies present in the `requirements.txt` file. It is not a part of the actual directory since it is part of the `gitignore`.
- `main.py` contains the core functionality of the project and brings all of the code in the `src/` directory together.
- `wifi_db/` contains the datasets provided for testing.
- `spec.pdf` contains the specification for the coursework, `report.pdf` contains the report.
- `img` contains all the images generated during profiling and testing. It can be overwritten for further testing as explained above, and gives a better visualization of the results obtained.
- `Clean Dataset Visualization.mp4` shows the tree generated using the entire clean data-set as the training set.

