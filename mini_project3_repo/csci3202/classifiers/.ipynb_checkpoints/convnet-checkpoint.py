import numpy as np

from csci3202.layers import *
from csci3202.fast_layers import *
from csci3202.layer_utils import *


class ConvNet(object):
  """
  A convolutional network with the following architecture:
  
  (conv - relu - 2x2 max pool)x3 - affine x 2- softmax
  
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
    # TODO: Initialize weights and biases for the convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #                                            
    ############################################################################
    C,H,W = input_dim
    W1 = np.random.normal(0,weight_scale,(num_filters,C,filter_size,filter_size))
    b1 = np.zeros((num_filters,))
    H1_conv_out = 1+(H-2)//2
    W1_conv_out = 1+(W-2)//2
    
    W2 = np.random.normal(0,weight_scale,(num_filters,num_filters,filter_size,filter_size))
    b2 = np.zeros((num_filters,))
    H2_conv_out = 1+(H1_conv_out-2)//2
    W2_conv_out = 1+(W1_conv_out-2)//2
    W3 = np.random.normal(0,weight_scale,(num_filters,num_filters,filter_size,filter_size))
    b3 = np.zeros((num_filters,))
    H3_conv_out = 1+(H2_conv_out-2)//2
    W3_conv_out = 1+(W2_conv_out-2)//2
    
    W4 = np.random.normal(0,weight_scale,(num_filters*H3_conv_out*W3_conv_out,hidden_dim))
    b4 = np.zeros((hidden_dim,))
    
    W5 = np.random.normal(0,weight_scale,(hidden_dim,num_classes))
    b5 = np.zeros((num_classes,))
    
    self.params['W1']=W1
    self.params['b1']=b1
    self.params['W2']=W2
    self.params['b2']=b2
    self.params['W3']=W3
    self.params['b3']=b3
    self.params['W4']=W4
    self.params['b4']=b4
    self.params['W5']=W5
    self.params['b5']=b5
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for theconvolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    W4, b4 = self.params['W4'], self.params['b4']
    W5, b5 = self.params['W5'], self.params['b5']
    
    # pass conv_param to the forward pass for the convolutional layer
    

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    # conv-relu-pool - 1
    filter_size = W1.shape[2]
    conv_param1 = {'stride': 1, 'pad': (filter_size - 1) // 2}
    conv1_out, conv1_cache = conv_forward_naive(X,W1,b1,conv_param1)
    cr1_out,cr1_cache=relu_forward(conv1_out)
    cpool1_out,cp1_cache= max_pool_forward_naive(cr1_out,pool_param)
    
    N,C,H,W = cpool1_out.shape
    bn_params1 = {'mode': 'train'}
    gamma1 = np.ones(C)
    beta1 = np.zeros(C)
    sbnorm1_out, sbnorm1_cache = spatial_batchnorm_forward(cpool1_out,gamma1,beta1,bn_params1)
    
    # conv-relu-pool - 2
    filter_size = W2.shape[2]
    conv_param2 = {'stride': 1, 'pad': (filter_size - 1) // 2}
    conv2_out, conv2_cache = conv_forward_naive(sbnorm1_out,W2,b2,conv_param2)
    cr2_out,cr2_cache=relu_forward(conv2_out)
    cpool2_out,cp2_cache= max_pool_forward_naive(cr2_out,pool_param)
    
    N,C,H,W = cpool2_out.shape
    bn_params2 = {'mode': 'train'}
    gamma2 = np.ones(C)
    beta2 = np.zeros(C)
    sbnorm2_out, sbnorm2_cache = spatial_batchnorm_forward(cpool2_out,gamma2,beta2,bn_params2)
    
    # conv-relu-pool - 3
    filter_size = W3.shape[2]
    conv_param3 = {'stride': 1, 'pad': (filter_size - 1) // 2}
    conv3_out, conv3_cache = conv_forward_naive(sbnorm2_out,W3,b3,conv_param3)
    cr3_out,cr3_cache=relu_forward(conv3_out)
    cpool3_out,cp3_cache= max_pool_forward_naive(cr3_out,pool_param)
    
    N,C,H,W = cpool3_out.shape
    bn_params3 = {'mode': 'train'}
    gamma3 = np.ones(C)
    beta3 = np.zeros(C)
    sbnorm3_out, sbnorm3_cache = spatial_batchnorm_forward(cpool3_out,gamma3,beta3,bn_params3)
    
    
    a1_out,a1_cache=affine_forward(sbnorm3_out,W4,b4)
    N,D = a1_out.shape
    bn_params4 = {'mode': 'train'}
    gamma4 = np.ones(D)
    beta4 = np.zeros(D)
    bnorm1_out,bnorm1_cache = batchnorm_forward(a1_out,gamma4,beta4,bn_params4)
    
    a2_out,a2_cache=affine_forward(bnorm1_out,W5,b5)
    N,D = a2_out.shape
    bn_params5 = {'mode': 'train'}
    gamma5 = np.ones(D)
    beta5 = np.zeros(D)
    scores,bnorm2_cache = batchnorm_forward(a2_out,gamma5,beta5,bn_params5)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss,dx_out = softmax_loss(scores,y)
    loss+= 0.5 * (np.sum(W1*W1)+np.sum(W2*W2)+np.sum(W3*W3)) * self.reg
    dx_a2, dgamma, dbeta = batchnorm_backward(dx_out,bnorm2_cache)
    dx_bnorm1, grads['W5'],grads['b5'] = affine_backward(dx_a2,a2_cache)
    
    dx_a1, dgamma, dbeta = batchnorm_backward(dx_bnorm1,bnorm1_cache)
    dx_sbnorm3,grads['W4'],grads['b4'] = affine_backward(dx_a1,a1_cache)
    
    
    dx_cpool3, dgamma, dbeta = spatial_batchnorm_backward(dx_sbnorm3,sbnorm3_cache)
    dx_cr3 = max_pool_backward_naive(dx_cpool3,cp3_cache)
    dx_conv3 = relu_backward(dx_cr3,cr3_cache)
    dx_sbnorm2,grads['W3'],grads['b3'] = conv_backward_naive(dx_conv3,conv3_cache)
    
    dx_cpool2, dgamma, dbeta = spatial_batchnorm_backward(dx_sbnorm2,sbnorm2_cache)
    dx_cr2 = max_pool_backward_naive(dx_cpool2,cp2_cache)
    dx_conv2 = relu_backward(dx_cr2,cr2_cache)
    dx_sbnorm1,grads['W2'],grads['b2'] = conv_backward_naive(dx_conv2,conv2_cache)
    
    dx_cpool1, dgamma, dbeta = spatial_batchnorm_backward(dx_sbnorm1,sbnorm1_cache)
    dx_cr1 = max_pool_backward_naive(dx_cpool1,cp1_cache)
    dx_conv1 = relu_backward(dx_cr1,cr1_cache)
    dx,grads['W1'],grads['b1'] = conv_backward_naive(dx_conv1,conv1_cache)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
