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
        # one_hot = features  XOR PROBLEM

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
    max_dev_accuracy = dev_accuracy = 0
    for I in xrange(num_iterations):
        cum_loss = 0.0 # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features) # convert features to a vector.
            #x = features XOR problem
            y = label                  # convert the label to number if needed.
            y = utils.L2I[y]
            loss, grads = mlp.loss_and_gradients(x,y,params)
            cum_loss += loss


            # YOUR CODE HERE
            # update the parameters according to the gradients
            # and the learning rate.

            for i in range(len(params)):
                eff_grad = grads[i] * learning_rate
                params[i] -= eff_grad

        #if ((I + 1) % 10 == 0): learning_rate /= 2
        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print (I, train_loss, train_accuracy, dev_accuracy)
        if (dev_accuracy  >= max_dev_accuracy) and (dev_accuracy > 0.6):
            max_dev_accuracy = dev_accuracy
            best_params = []
            best_params_index = I
            for i in range(len(params)):
                best_params.append(np.copy(params[i]))


    if (max_dev_accuracy > 0.6):
        train_accuracy = accuracy_on_dataset(train_data, best_params)
        dev_accuracy = accuracy_on_dataset(dev_data, best_params)
        print ("best_params")
        print (" train_accuracy, dev_accuracy")
        print (train_accuracy, dev_accuracy)
        return best_params
    else:
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

    out_dim = len(utils.L2I)
    in_dim = len(utils.vocab)
    learning_rate = 6.46428571e-04
    #from xor_data import data
    #train_data = data XOR problem
    #dev_data = data XOR problem
    params = mlp.create_classifier(in_dim, hid_dim=15,out_dim = out_dim)
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)

    # predict on test
    test_data = utils.read_data("test")
    f = open("test.pred", "w+")
    inv_map = {v: k for k, v in utils.L2I.iteritems()}
    index = 0
    for label, features in test_data:
        # YOUR CODE HERE
        # Compute the accuracy (a scalar) of the current parameters
        # on the dataset.
        index += 1
        #label = utils.L2I[label]
        one_hot = feats_to_vec(features)

        predicted_label = mlp.predict(one_hot,trained_params)
        f.write("%s\t%s" % ((inv_map[predicted_label]),features))
        if (index != 300):
            f.write("\n")