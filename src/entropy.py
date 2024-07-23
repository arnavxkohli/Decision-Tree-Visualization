import numpy as np


class InformationGain:
    """
    Encapsulation of all the functions related to calculating the
    information gain to determine the optimum split attribute and value.
    """
    def __init__(self, y: np.ndarray, x: np.ndarray):
        """
        Constructor

        Args:
            y (np.ndarray): y is a numpy array with shape (N, ),
            where N is the number of instances; the values range from 1 to 4
            and represent the rooms.
            x (np.ndarray): x is a numpy array with shape (N, K),
            where N is the number of instances; K is the number of
            features or attributes.
        """
        self.x = x
        self.y = y

    def calculate_entropy(self, values: np.ndarray) -> float:
        """
        For a given numpy array containing the labels, returns the entropy
        associated with it.

        Args:
            values (np.ndarray): Numpy array containing the labels associated
            with the attribute.

        Returns:
            float: The calculated entropy value.
        """
        total_values = values.size
        # find the unique labels and their counts
        unique, count = np.unique(values, return_counts = True)
        # convert them to a dictionary
        values_dict = dict(zip(unique, count))
        # initialize the entropy variable as 0
        entropy = 0
        # loop through each unique label
        for value in values_dict:
            # find the probability for each label based on the entire set of
            # values passed, and after converting it to entropy, add it to
            # the net entropy
            probability = values_dict[value]/total_values
            entropy += -(probability * np.log2(probability))
        # return the summed entropies
        return entropy

    def calculate_information_gains(self, attribute_values: np.ndarray, \
                                    label_values: np.ndarray) -> dict:
        """
        For a given set of sorted attribute values and their corresponding
        labels, returns a dictionary containing the mean between each pair of
        values, and their associated information gains.

        Args:
            attribute_values (np.ndarray): Array of sorted values for one
            particular attribute.

        Returns:
            dict: Dictionary containing the means and their corresponding
            information gains.
        """
        # initialize the information gains as a dictionary
        information_gains = {}
        # find entropy of the parent node
        root_entropy = self.calculate_entropy(self.y)
        # loop through all the attribute values provided, except the last one,
        # we want to step taking two attribute values at once to calculate
        # the mean between them
        for i in range(attribute_values.size - 1):
            # calculate the mean of any two consecutive attribute values
            mean = (attribute_values[i] + attribute_values[i+1])/2
            # left split
            left_indices = attribute_values < mean
            # right split
            right_indices = attribute_values >= mean
            # scaled left entropy, takes into account the cardinalities
            left_entropy = (((self.calculate_entropy(label_values \
                              [left_indices]))/attribute_values.size) * \
                              attribute_values[left_indices].size)
            # scaled right entropy, takes into account the cardinalities
            right_entropy = (((self.calculate_entropy(label_values \
                               [right_indices]))/attribute_values.size) \
                               * attribute_values[right_indices].size)
            # memoisation step, perform check to see if the information gain
            # has already been calculated for the calculated mean, and if so
            # then do not repeat calculation
            if str(mean) not in information_gains:
                information_gains[str(mean)] = root_entropy - (left_entropy + \
                                                               right_entropy)
        # return the information gains dictionary
        return information_gains

    def find_split(self) -> tuple[int, float]:
        """
        Iterates through each attribute in the dataset, sorting the values
        and maintaining ordering with their corresponding labels. Finds the
        splitting attribute and splitting value by maximizing information gain.

        Returns:
            tuple[int, float]: Returned tuple containing the splitting
            attribute (the column with maximum entropy) and the splitting
            value (the value within the attribute set that maximizes
            information gain).
        """
        # find number of columns (corresponding to the attributes) using shape
        # of numpy array
        columns = self.x.shape[1]
        # initialize return variables
        returned_attribute = 0
        returned_value = 0
        # initialize global maximum information gain as -inf, to be
        # overwritten
        max_information_gain = -float('inf')
        # loop through each column, basically perform for each attribute
        for column in range(columns):
            # return an array of indices that would sort the attribute, needed
            # to preserve the order of attributes to labels
            sorted_indices = np.argsort(self.x[:, column])
            # get attributes in sorted order
            attribute_values = self.x[sorted_indices, column]
            # get their corresponding labels
            label_values = self.y[sorted_indices]
            # return a dictionary containing all the information gains
            # extracted from the information obtained above
            information_gains = \
                self.calculate_information_gains(attribute_values = \
                                                 attribute_values, \
                                                 label_values = label_values)
            # for each information gain returned, check if greater than
            # the global maximum, if so then replace with global maximum
            # and set return variables accordingly
            for value in information_gains:
                if information_gains[value] > max_information_gain:
                    max_information_gain = information_gains[value]
                    returned_attribute = column
                    returned_value = float(value)

        return returned_attribute, returned_value
