# A sample training component that trains a simple scikit-learn decision tree model.
# This implementation works in File mode and makes no assumptions about the input file names.
# Input is specified as CSV with a data point in each row and the labels in the first column.

from __future__ import print_function

import os
import pickle

import pandas as pd
import sagemaker_containers
from sklearn import tree

# This algorithm has a single channel of input data called 'training'. Since we run in
# File mode, the input files are copied to the directory specified here.
channel_name = 'training'


# The function to execute the training.
def main():
    env = sagemaker_containers.training_env()

    print('Starting the training.')
    training_path = env.channel_input_dirs[channel_name]

    # Take the set of files and read them all into a single pandas dataframe
    input_files = [os.path.join(training_path, file) for file in os.listdir(training_path)]
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(training_path, channel_name))
    raw_data = [pd.read_csv(file, header=None) for file in input_files]
    train_data = pd.concat(raw_data)

    # labels are in the first column
    train_y = train_data.ix[:, 0]
    train_X = train_data.ix[:, 1:]

    # Here we only support a single hyperparameter. Note that hyperparameters are always passed in as
    # strings, so we need to do any necessary conversions.
    max_leaf_nodes = env.hyperparameters.get('max_leaf_nodes', None)
    if max_leaf_nodes is not None:
        max_leaf_nodes = int(max_leaf_nodes)

    # Now use scikit-learn's decision tree classifier to train the model.
    clf = tree.DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes)
    clf = clf.fit(train_X, train_y)


    # save the model
    with open(os.path.join(env.model_dir, 'decision-tree-model.pkl'), 'w') as out:
        pickle.dump(clf, out)
    print('Training complete.')


if __name__ == '__main__':
    main()
