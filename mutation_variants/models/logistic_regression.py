from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
import torch.optim as optim
import pylab as pp




class LogisticRegression(object):
  def __init__(self, dim, l1 = 0.0, l2 = 0.0 ):
    super(LogisticRegression, self).__init__()
    self.dim       = dim
    self.l1        = l1
    self.l2        = l2

    self.model = torch.nn.Sequential()
    self.model.add_module("linear", torch.nn.Linear(self.dim, 1, bias=True))
    
    self.log_activation = torch.nn.LogSigmoid()
    self.activation = torch.nn.Sigmoid()
    
    self.P = []
    for p in self.model.parameters():
      self.P.append(p)
    self.w = self.P[0]
    #self.model.add_module("sigmoid", torch.nn.Sigmoid )
    #m = nn.LogSoftmax()
    #    >>> loss = nn.NLLLoss()
  
  def reset(self):
    for p in self.model.parameters():
      p *= 0
              
  def forward( self, X ):
    return torch.nn.LogSigmoid( self.model.forward(X) ), self.model.forward(X)-torch.nn.LogSigmoid( self.model.forward(X) ) #torch.mv(X,self.w) + self.bias.expand(X.size()[0])

  def predict( self, X_val ):
    X = Variable(X_val, requires_grad=False)
    return self.activation( self.model.forward(X) )
    
  def fit( self, X_val, y_val, n_epochs = 10, lr = 2*1e-2, logging_frequency = 1, normalize=False ):
    X = Variable(X_val, requires_grad=False)
    y = Variable(y_val, requires_grad=False)
    #data  = [X, y]

    optimizer = optim.Adam(self.model.parameters(), lr=lr)
    #entropy_loss = torch.nn.BCELoss(size_average=True)
    
    for epoch in xrange(1, n_epochs):
      #self.train()
      train_loss = 0
      
      optimizer.zero_grad()
      
      wx = self.model(X)
      log_p_1 = self.log_activation(wx)
      log_p_2 = -wx+log_p_1
      #print fx
      data_cost = -torch.mean( y*log_p_1 + (1-y)*log_p_2 )
      
      weight_cost = 0
      if self.l1 > 0:
        weight_cost += self.l1*torch.sum( torch.abs( self.w ) )
      if self.l2 > 0:
        weight_cost += self.l2*torch.sum( torch.pow( self.w, 2 ) )
        #loss += l1*torch.sum( torch.abs( self.alpha) )
      loss = data_cost + weight_cost
      loss.backward()
      optimizer.step()
      
      if epoch%logging_frequency == 0:
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, data_cost.data[0] ))
    if epoch%logging_frequency == 0:
      print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, data_cost.data[0] ))
#
#
# if __name__ == '__main__':
#   l1 = 0.1
#   n = 500
#   dim = 3
#
#   w = np.random.randn(3)
#   w[1] = 0
#   b = 0.5
#   noise = 0.25
#   X = np.random.randn( n,dim )
#   y = np.dot( X, w ) + b + noise*np.random.randn(n)
#
#   from lifelines import datasets
#   data=datasets.load_regression_dataset()
#   #
#   X = Variable( torch.FloatTensor( X ) )
#   y = Variable( torch.FloatTensor( y ) )
#
#   model = PytorchLasso( dim, l1 )
#   #data  = [var_E, var_T, var_Z]
#
#   #def loss_function( log_hazard, log_survival):
#   #  return -torch.sum( log_hazard + log_survival )
#
#   #optimizer = optim.Adam(model.parameters(), lr=0.01)
#
#   model.fit( X,y )
#   y_pred = model.predict( X )
#
#   pp.figure()
#   pp.subplot(1,2,1)
#   pp.plot( w, 'bo-', alpha=0.5)
#   pp.plot( model.w.data.numpy(), 'ro-', alpha=0.5)
#   pp.subplot(1,2,2)
#   pp.plot( y.data.numpy(), y_pred.data.numpy(), 'bo', alpha=0.5)
#   pp.axis('equal')
#   pp.show()

  