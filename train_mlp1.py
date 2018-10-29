import mlp1 as mlp
import random
import train_loglin
import utils
import numpy as np
STUDENT={'name': 'Ofer Sabo',
         'ID': '201511110'}

def feats_to_vec(features):
    # YOUR CODE HERE.
    # Should return a numpy vector of features.
    return train_loglin.feats_to_vec(features)

def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        # YOUR CODE HERE
        # Compute the accuracy (a scalar) of the current parameters
        # on the dataset.
        # accuracy is (correct_predictions / all_predictions)
        label = utils.L2I[label]
        one_hot = feats_to_vec(features)
        predicted_label = mlp.predict(one_hot,params)
        if (predicted_label != label):
            bad += 1
        else:
            good += 1

    return good / (good + bad)

def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    for I in xrange(num_iterations):
        cum_loss = 0.0 # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features) # convert features to a vector.
            y = label                  # convert the label to number if needed.

            loss, grads = mlp.loss_and_gradients(x,utils.L2I[y],params)
            cum_loss += loss
            # YOUR CODE HERE
            # update the parameters according to the gradients
            # and the learning rate.
            for i in range(len(params)):
                #print (np.max(grads[i]))
                params[i] -= grads[i] * learning_rate

        if ((I + 1) % 40 == 0): learning_rate /= 2
        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print I, train_loss, train_accuracy, dev_accuracy
    return params

if __name__ == '__main__':
    # YOUR CODE HERE
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.
    
    # ...
    train_data = utils.read_data("train")
    y_train = [l[0] for l in train_data]
    X_trrain = [l[1] for l in train_data]
    dev_data = utils.read_data("dev")
    y_dev = [l[0] for l in dev_data]
    X_dev = [l[1] for l in dev_data]
    num_iterations = 100
    learning_rate = 1.2
    out_dim = len(utils.L2I)
    in_dim = len(utils.vocab)
   
    params = mlp.create_classifier(in_dim, hid_dim=10,out_dim=out_dim)
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)

