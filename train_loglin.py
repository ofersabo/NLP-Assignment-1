# -*- coding: utf-8 -*-
import loglinear as ll
import random
from utils import *
import numpy as np
start_count = False
problematic_line = []
STUDENT={'name': 'Ofer Sabo',
         'ID': '201511110'}

def feats_to_vec(features):
    # YOUR CODE HERE.
    # Should return a numpy vector of features.
    numpy_feature = list()
    one_hot = np.zeros(len(vocab))
    bi_gram = text_to_bigrams(features)
    for bi in bi_gram:
        if bi in vocab:
            one_hot[F2I[bi]] = 1

    return one_hot

def accuracy_on_dataset(dataset, params):
    global problematic_line
    from loglinear import predict
    problematic_line = []
    good = bad = 0.0
    for label, features in dataset:
        # YOUR CODE HERE
        # Compute the accuracy (a scalar) of the current parameters
        # on the dataset.
        # accuracy is (correct_predictions / all_predictions)
        label = L2I[label]
        one_hot = feats_to_vec(features)
        predicted_label = predict(one_hot,params)
        if (predicted_label != label):
            bad += 1
            if (start_count):
                problematic_line.append( features)
        else:
            good += 1

    problematic_line = set(problematic_line)
    return good / (good + bad)

def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    """
    Create and train.txt a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    global start_count
    for I in xrange(num_iterations):
        cum_loss = 0.0 # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features) # convert features to a vector.
            y = L2I[label]             # convert the label to number if needed.
            loss, grads = ll.loss_and_gradients(x,y,params)
            cum_loss += loss
            # YOUR CODE HERE
            # update the parameters according to the gradients
            # and the learning rate.
            params[0] -= grads[0] * learning_rate
            params[1] -= grads[1] * learning_rate

        if ((I+1)%10 == 0 ): learning_rate /= 10
        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        if (dev_accuracy > 0.88): start_count = True
        print I, train_loss, train_accuracy, dev_accuracy
    return params

if __name__ == '__main__':
    # YOUR CODE HERE
    # write code to load the train.txt and dev sets, set up whatever you need,
    # and call train_classifier.
    
    # ...
    # #
    # with open("train.txt.txt") as fp:
    #     for i, line in enumerate(fp):
    #         if "\xe2" in line:
    #             print i, repr(line)
    train_data = read_data("train")
    y_train = [l[0] for l in train_data]
    X_trrain = [l[1] for l in train_data]
    dev_data = read_data("dev")
    y_dev = [l[0] for l in dev_data]
    X_dev = [l[1] for l in dev_data]
    num_iterations =100
    learning_rate = 1e-4
    out_dim = len(L2I)
    in_dim = len(vocab)
    params = ll.create_classifier(in_dim, out_dim)
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)

    print problematic_line
