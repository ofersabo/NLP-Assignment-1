import numpy as np
STUDENT={'name': 'Ofer Sabo',
         'ID': '201511110'}

def classifier_output(x, params):
    # YOUR CODE HERE.
    post_activation = []
    main_input = (np.array(x)[np.newaxis, :])
    num_tanh_layers = (len(params) - 2) / 2
    for i in range(num_tanh_layers):
        if i == 0:
            layer_input = main_input
        else:
            layer_input = post_activation[-1]
        result_after_tanh = go_through_tanh(layer_input, params[2 * i], params[2 * i + 1])
        post_activation.append(result_after_tanh)
    probs = soft_max_function_result(post_activation[-1], params[-2:])
    return probs


def predict(x, params):
    return np.argmax(classifier_output(x, params))


def go_through_tanh(x,W,b):
    multi = np.dot(x, W)
    output = multi + b
    return np.tanh(output)


def soft_max_function_result(x,params):
    W,b = params
    multi = np.dot(x,W)
    score = multi.flatten() + b
    score = score - np.max(score)
    probs = np.exp(score)
    suming = np.sum(probs)
    return_value = probs / suming
    return return_value

def loss_and_gradients(x, y, params):
    """
    params: a list as created by create_classifier(...)

    returns:
        loss,[gW1, gb1, gW2, gb2, ...]

    loss: scalar
    gW1: matrix, gradients of W1
    gb1: vector, gradients of b1
    gW2: matrix, gradients of W2
    gb2: vector, gradients of b2
    ...

    (of course, if we request a linear classifier (ie, params is of length 2),
    you should not have gW2 and gb2.)
    """
    # YOU CODE HERE
    post_activation = []
    result_at_layer = []
    main_input = (np.array(x)[np.newaxis,:])
    num_tanh_layers = (len(params)-2)/2
    for i in range(num_tanh_layers):
        if i == 0: layer_input = main_input
        else: layer_input = post_activation[-1]
        result_after_tanh = go_through_tanh(layer_input,params[2*i],params[2*i+1])
        post_activation.append(result_after_tanh)
    soft_max_result = soft_max_function_result(post_activation[-1],params[-2:])
    loss = - np.log(soft_max_result[y])
    gW = np.zeros_like(params[-2])
    gb = np.zeros_like(params[-1])
    gW += soft_max_result
    gb += soft_max_result
    gW[:, y] -= 1
    gb[y] -= 1
    gW = (post_activation[-1].T * gW)

    grads = []
    grads.append(gb)
    grads.append(gW)
    for i in range(num_tanh_layers):
        db_next = grads[2*i]
        #dW_next = grads[2*i+1]
        x_next  = post_activation[-(i+1)]
        if i == num_tanh_layers-1:
            x_pre = main_input
        else:
            x_pre   = post_activation[-(i+2)]
        W       = params[-(2*i+2)]
        dout    = np.dot(db_next,W.T)
        db_pre  = (np.ones_like(x_next) - (x_next ** 2)) * dout
        dW_pre  =  np.dot(x_pre.T,db_pre)
        #dout_pre = db_pre * W.T
        grads.append(db_pre.flatten())
        grads.append(dW_pre)

    return loss,grads[::-1]





def create_classifier(dims):
    """
    returns the parameters for a multi-layer perceptron with an arbitrary number
    of hidden layers.
    dims is a list of length at least 2, where the first item is the input
    dimension, the last item is the output dimension, and the ones in between
    are the hidden layers.
    For example, for:
        dims = [300, 20, 30, 40, 5]
    We will have input of 300 dimension, a hidden layer of 20 dimension, passed
    to a layer of 30 dimensions, passed to learn of 40 dimensions, and finally
    an output of 5 dimensions.
    
    Assume a tanh activation function between all the layers.

    return:
    a flat list of parameters where the first two elements are the W and b from input
    to first layer, then the second two are the matrix and vector from first to
    second layer, and so on.
    """
    params = []
    for i in range(len(dims)-1):
        W = np.random.rand(dims[i],dims[i+1]) * (2.0/float(dims[i]+dims[i+1]))
        b = np.random.rand(dims[i+1]) * (1.0/float(dims[i+1]))
        params.append((W))
        params.append((b))
    return params


if __name__ == '__main__':
    # Sanity checks for mlp1. If these fail, your softmax is definitely wrong.
    # If these pass, it may or may not be correct.

    # Sanity checks. If these fail, your gradient calculation is definitely wrong.
    # If they pass, it is likely, but not certainly, correct.
    from grad_check import gradient_check

    params = create_classifier([3,40,100,2,50,10,4])

    for i in range(len(params)):
        print (params[i].shape)


    def _loss_and_grad_for_check(parameter):
        global params
        #print (params[2])
        params[index] = parameter
        loss,grads = loss_and_gradients([1,2,3],0,params)
        return loss,grads[index]


    for _ in xrange(100):
        for i in range(len(params)):
            index = i
            parametr = np.random.random_sample(params[i].shape)
            if (not gradient_check(_loss_and_grad_for_check, parametr)):
                print ("ERROR")
                exit()

