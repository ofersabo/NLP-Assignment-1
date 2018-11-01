import numpy as np

from loglinear import softmax
STUDENT={'name': 'Ofer Sabo',
         'ID': '201511110'}

def classifier_output(x, params):
    # YOUR CODE HERE.
    probs = 0
    return probs

def predict(x, params):
    return np.argmax(classifier_output(x, params))

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
    pre_activation  = []
    result_at_layer = []
    post_activation.append(x)
    pre_activation.append(None)
    num_hidden_layers = len(params)/2 - 1
    for i in range(num_hidden_layers):
        output = post_activation[2*i] * params[2*i] + params[2*i+1]
        pre_activation.append(np.copy(output))
        post_activation.append(np.tanh(output))

    probs = softmax(post_activation[-1])
    loss = - np.log(probs[y])
    grads = []
    gW = np.zeros_like(W)
    gb = np.zeros_like(b)
    gW += probs
    gb += probs
    gW /= np.sum(probs)
    gb /= np.sum(probs)
    gW[:, y] -= 1
    gb[y] -= 1
    gW = (x.T * gW)
    grads.append(gb)
    grads.append(gW)
    for i in range(num_hidden_layers):
        db_next = grads[2*i]
        dW_next = grads[2*i+1]
        z = post_activation[num_hidden_layers - i + 1]
        q = post_activation[num_hidden_layers - i ]
        dZ = np.dot(db_next,np.matrix(dW_next).T)
        db_pre = (np.ones_like(q.shape[1]) - (z ** 2)) * dZ
        dW_pre 

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

    params = create_classifier([3,10,4])

    W, b, U, b_tag = params

    for i in range(len(params)):
        print params[i].shape

    def _loss_and_W_grad(W):
        global U
        global b
        global b_tag
        loss,grads = loss_and_gradients([1,2,3],0,[W, b, U, b_tag])
        return loss,grads[0]

    def _loss_and_b_grad(b):
        global U
        global W
        global b_tag
        loss,grads = loss_and_gradients([1,2,3],0,[W, b, U, b_tag])
        return loss,grads[1]

    def _loss_and_U_grad(U):
        global b_tag
        global W
        global b
        loss,grads = loss_and_gradients([1,2,3],0,[W, b, U, b_tag])
        return loss,grads[2]

    def _loss_and_b_tag_grad(b_tag):
        global U
        global W
        global b
        loss,grads = loss_and_gradients([1,2,3],0,[W, b, U, b_tag])
        return loss,grads[3]

    for _ in xrange(100):
        U = np.random.randn(U.shape[0],U.shape[1])
        b_tag = np.random.randn(b_tag.shape[0])
        b = np.random.randn(b.shape[0])
        if (not gradient_check(_loss_and_b_grad, b)):
            print "ERROR"
            exit()
        if (not gradient_check(_loss_and_W_grad, W)):
            print "ERROR"
            exit()
        if (not gradient_check(_loss_and_U_grad, U)):
            print "ERROR"
            exit()
        if (not gradient_check(_loss_and_b_tag_grad, b_tag)):
            print "ERROR"
            exit()


