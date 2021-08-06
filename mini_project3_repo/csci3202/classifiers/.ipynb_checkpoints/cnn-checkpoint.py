import numpy as np

from csci3202.layers import *
from csci3202.fast_layers import *
from csci3202.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C,H,W = input_dim
    W1 = np.random.normal(0,weight_scale,(num_filters,C,filter_size,filter_size))
    b1 = np.zeros((num_filters,))
    
    H_conv_out = 1+(H-2)//2
    W_conv_out = 1+(W-2)//2
    
    W2 = np.random.normal(0,weight_scale,(num_filters*H_conv_out*W_conv_out,hidden_dim))
    b2 = np.zeros((hidden_dim,))
    
    W3 = np.random.normal(0,weight_scale,(hidden_dim,num_classes))
    b3 = np.zeros((num_classes,))
    
    self.params['W1']=W1
    self.params['b1']=b1
    self.params['W2']=W2
    self.params['b2']=b2
    self.params['W3']=W3
    self.params['b3']=b3
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    conv_out, conv_cache = conv_forward_naive(X,W1,b1,conv_param)
    cr_out,cr_cache=relu_forward(conv_out)
    cpool_out,cp_cache= max_pool_forward_naive(cr_out,pool_param)
    N,C,H,W = cpool_out.shape
    bn_params = {'mode': 'train'}
    gamma = np.ones(C)
    beta = np.zeros(C)
    sbnorm_out, sbnorm_cache = spatial_batchnorm_forward(cpool_out,gamma,beta,bn_params)
    aHidden_out,aHidden_cache=affine_forward(sbnorm_out,W2,b2)
    arHidden_out,arHidden_cache=relu_forward(aHidden_out)
    N,D = arHidden_out.shape
    bn_params1 = {'mode': 'train'}
    gamma1 = np.ones(D)
    beta1 = np.zeros(D)
    bnorm1_out,bnorm1_cache = batchnorm_forward(arHidden_out,gamma1,beta1,bn_params1)
    aout,aout_cache=affine_forward(bnorm1_out,W3,b3)
    N,D1 = aout.shape
    bn_params2 = {'mode': 'train'}
    gamma2 = np.ones(D1)
    beta2 = np.zeros(D1)
    scores,bnorm2_cache = batchnorm_forward(aout,gamma2,beta2,bn_params2)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss,dx_out = softmax_loss(scores,y)
    loss+= 0.5 * (np.sum(W1*W1)+np.sum(W2*W2)+np.sum(W3*W3)) * self.reg
    dx_bnorm2, dgamma, dbeta = batchnorm_backward(dx_out,bnorm2_cache)
    dx_arHidden, grads['W3'],grads['b3'] = affine_backward(dx_bnorm2,aout_cache)
    dx_bnorm1, dgamma, dbeta = batchnorm_backward(dx_arHidden,bnorm1_cache)
    dx_aHidden = relu_backward(dx_bnorm1,arHidden_cache)
    dx_sbnorm,grads['W2'],grads['b2'] = affine_backward(dx_aHidden,aHidden_cache)
    dx_cpool, dgamma, dbeta = spatial_batchnorm_backward(dx_sbnorm,sbnorm_cache)
    dx_cr = max_pool_backward_naive(dx_cpool,cp_cache)
    dx_conv = relu_backward(dx_cr,cr_cache)
    dx,grads['W1'],grads['b1'] = conv_backward_naive(dx_conv,conv_cache)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
