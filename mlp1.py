import numpy as np

STUDENT={'name': 'YOUR NAME',
         'ID': 'YOUR ID NUMBER'}

def classifier_output(x, params):
    # YOUR CODE HERE.
    W, b, U, b_tag  = params
    post_activation = get_post_activation(x,params)
    post_second_layer = np.dot(post_activation,U) + b_tag
    from loglinear import softmax
    probs = softmax(post_second_layer)
    return probs

def get_post_activation(x,params):
    W, b, U, b_tag  = params
    x = np.array(x)
    x = x[np.newaxis,:]
    multi = np.dot(x,W)
    post_first_layer = multi.flatten() + b
    post_activation = np.tanh(post_first_layer)
    return post_activation

def predict(x, params):
    """
    params: a list of the form [W, b, U, b_tag]
    """
    return np.argmax(classifier_output(x, params))

def loss_and_gradients(x, y, params):
    """
    params: a list of the form [W, b, U, b_tag]

    returns:
        loss,[gW, gb, gU, gb_tag]

    loss: scalar
    gW: matrix, gradients of W
    gb: vector, gradients of b
    gU: matrix, gradients of U
    gb_tag: vector, gradients of b_tag
    """
    W, b, U, b_tag  = params

    # YOU CODE HERE
    post_activation = get_post_activation(x,params)
    probs = classifier_output(x,params)
    loss = - np.log(probs[y])
    import loglinear
    ___ , g_activation = loglinear.loss_and_gradients(post_activation,y,[U,b_tag])
    gU = g_activation[0]
    gb_tag = g_activation[1]
    g_post_tanh = np.dot(gb_tag,U.T)
    grad_pre_tanh = (np.ones_like(post_activation)-(post_activation**2)) * g_post_tanh

    gb = grad_pre_tanh

    grad_pre_tanh_numpy =  grad_pre_tanh[np.newaxis,:]

    x_input = np.array(x)
    x_input = x_input[:,np.newaxis]
    gW = np.dot(x_input,grad_pre_tanh_numpy)
    #loss += (np.sum(W**2) + np.sum(b**2) + np.sum(U**2) + np.sum(b_tag**2)) ** 2
    return loss,[gW,gb,gU,gb_tag]





def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.

    return:
    a flat list of 4 elements, W, b, U, b_tag.
    """
    W = np.random.rand(in_dim,hid_dim) * (2.0/float(in_dim+hid_dim))
    b = np.random.rand(hid_dim) * (1.0/float(hid_dim))
    U = np.random.rand(hid_dim,out_dim) * (2.0/float(out_dim+hid_dim))
    b_tag = np.random.rand(out_dim) * (1.0/float(out_dim))
    params = [W,b,U,b_tag]
    return params


if __name__ == '__main__':
    # Sanity checks for mlp1. If these fail, your softmax is definitely wrong.
    # If these pass, it may or may not be correct.

    # Sanity checks. If these fail, your gradient calculation is definitely wrong.
    # If they pass, it is likely, but not certainly, correct.
    from grad_check import gradient_check

    W, b, U, b_tag = create_classifier(3,10,4)

    params = [W, b, U, b_tag]

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


