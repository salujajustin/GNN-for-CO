# -*- coding: utf-8 -*-
"""Attention Test
#Author - Everett Knag

This code is based on the attention learn to solve paper but has been heavily modified

In particular, everything has been highly vectorized for more speeed.

It is still a work in progress. The model architecture is almost complete, but is missing the following items:
  - The decoder only has a single 1-head attention output layer, it still needs the 8 head layer as well
  - The decoder does not apply a mask over already selected nodes as is stated in the paper.
  - No correctness checking or testing has been performed
  - The loss function (sum of path weights) has not been implemented
  - No optimizer has been added
  - The process would very seriously benefit from a sparse attention mechanism 
    (or one that considers at most K other nodes).

There is some hope that we have a speedup here, although we can't be sure until it is fully implemented.
Run time on T4 GPU is about 2 min/epoch for n = 20, batch_size = 512.

Previously, there were high memory usage issues, but these seem to have mysteriously
gone away
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import gc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def memory_usage():
  return torch.cuda.memory_summary(device)

def check_shape(A,values):
  for i,v in enumerate(values):
    if (i>= len(A.shape)) or (A.shape[i] != v):
      print("dimensions disagree at index %d" % i)
      print("Array shape is ", A.shape)
      print("assumed shape is " + str(values))



class attention(nn.Module):
  def __init__(self, M, d_h, d_k, d_v, tanh=False, C=10):
    super().__init__()

    d_q = d_k
    
    self.d_q = d_q
    self.d_h = d_h
    self.d_k = d_k
    self.d_v = d_v
    self.d_q = d_q
    self.sqrt_d_k = d_k**0.5

    self.tanh = tanh
    self.C = C

    self.W_Q = nn.Parameter(torch.empty(size=(M, d_h, d_q)))
    self.W_K = nn.Parameter(torch.empty(size=(M, d_h, d_k)))
    self.W_V = nn.Parameter(torch.empty(size=(M, d_h, d_v)))
    self.W_O = nn.Parameter(torch.empty(size=(M, d_v, d_h)))

    nn.init.xavier_uniform_(self.W_Q.data, gain=1.414)
    nn.init.xavier_uniform_(self.W_K.data, gain=1.414)
    nn.init.xavier_uniform_(self.W_V.data, gain=1.414)
    nn.init.xavier_uniform_(self.W_O.data, gain=1.414)


  def forward(self, H):
    if(len(H.shape) == 3):
      batch_size = H.shape[0]
      n = H.shape[1]
      d_h = H.shape[2]
      H = H.view(batch_size, 1, n, d_h)
    
    Q = H @ self.W_Q
    K = H @ self.W_K

    U = 1/self.sqrt_d_k * Q @ torch.transpose(K, -1, -2)

    del Q
    del K
    #make sure dim one is correct
    if self.tanh:
      assert(len(U.shape)==2)
      return torch.sofmax(self.C * torch.tanh(U), dim = -1)
    A = torch.softmax(U, dim = -1)

    del U

    V = H @ self.W_V

    H_prime = A @ V

    del A, V

    H = torch.sum(H_prime @ self.W_O, dim = -3)
    del H_prime

    torch.cuda.empty_cache()

    H = H.view(batch_size, n, d_h)



    return H

class out_attention(nn.Module):
  def __init__(self, d_h, d_k, C=10):
    super().__init__()

    #h_c = \overline(H) || h_{\pi_{t-1}} || h_{\pi_1} 
    self.d_c = 3*d_h 
    self.d_q = d_k
    self.d_h = d_h
    self.d_k = d_k
    self.sqrt_d_k = d_k**0.5

    self.C = C

    self.W_Q = nn.Parameter(torch.empty(size=(self.d_c, d_k)))
    self.W_K = nn.Parameter(torch.empty(size=(d_h, d_k)))

    nn.init.xavier_uniform_(self.W_Q.data, gain=1.414)
    nn.init.xavier_uniform_(self.W_K.data, gain=1.414)


  def forward(self, h_c, H):
    B = H.shape[0]
    ##h_c size (B, 1, 3*d_h) 
    ##H - (B, 100, d_h)

    #check_shape(h_c, [B, 1 , self.d_c])
    #check_shape(H, [B, n, self.d_h])

    Q = h_c @ self.W_Q
    K = H @ self.W_K

    #This takes a crazy amount of memory
    #https://discuss.pytorch.org/t/memory-inefficient-in-batch-matrix-multiplication-with-autograd/28164/3
    #might want to consider a different operation rather than matmul
    U = 1/self.sqrt_d_k * Q @ torch.transpose(K, -1, -2)

    U = torch.squeeze(U)
    del Q
    del K
    

    return torch.softmax(self.C * torch.tanh(U), dim = -1)

class normalization(nn.Module):
  def __init__(self, d_h, n):
    super().__init__()
    self.d_h = d_h

    #self.normalizer = nn.BatchNorm1d(d_h)
    #consider nn.LayerNorm(d_h)(Q) for speed?
    #or using d_h * n and a flatten
    self.normalizer = nn.BatchNorm1d(d_h*n)

  def forward(self, H):

    (bs, n, d_h) = H.shape

    ##we don't want to zero everything :(!
    if(bs <= 2):
      return H
    return self.normalizer(H.view(bs,-1)).view(bs, n, d_h)
    #return torch.stack([self.normalizer(h_i) for h_i in torch.unbind(H, dim=-2)], dim=-2)



class encoder_unit(nn.Module):
    def __init__(self, M, n, d_h, d_k, d_v, num_hidden_units, share_params=False, batch_norm=True):
      super().__init__()
      self.M = M
      self.d_h = d_h

      self.share_params = share_params
      self.batch_norm = batch_norm

      self.attention = attention(M, d_h, d_k, d_v)


      self.linear = nn.Sequential(nn.Linear(d_h,num_hidden_units), \
                                  nn.ReLU(), \
                                  nn.Linear(num_hidden_units, d_h))
      
      self.BN_1 = normalization(d_h, n) ##should this be 2d?
      self.BN_2 = normalization(d_h, n) ##should this be 2d?

      """
      self.BN_1 = nn.LayerNorm([n,d_h])
      self.BN_2 = nn.LayerNorm([n,d_h])
      """

    def forward(self, H):

      if self.batch_norm:
        H_hat = self.BN_1(H + self.attention(H))
        H = self.BN_2(H_hat + self.linear(H_hat))
      else:
        H_hat = H + self.attention(H)
        H = (H_hat + self.linear(H_hat))

      return H

class encoder(nn.Module):
  def __init__(self, T, M, n, d_x, d_h, d_k, d_v, num_hidden_units, share_params=False,\
               batch_norm=True):
    super().__init__()
    self.T = T
    self.num_iters = T
    self.M = M
    self.d_h = d_h

    self.share_params = share_params
    self.batch_norm = batch_norm

    if share_params:
        self.T = 1
    self.to_hidden = nn.Linear(d_x, d_h)

    self.layers = nn.ModuleList([encoder_unit(M, n, d_h, d_k, d_v,\
                                              num_hidden_units, \
                                              share_params, \
                                              batch_norm) \
                   for t in range(self.T)])
  def forward(self, X):
    
    H = self.to_hidden(X)
    for t in range(self.num_iters):
      
      i = t % self.T
      H = self.layers[i](H)
      torch.cuda.empty_cache()

    return H

class decoder_unit(nn.Module):
  #decoder takes in h_c,t and returns some probability for P(\pi_t)
  def __init__(self, d_h):
    super().__init__()
    self.d_h = d_h
    self.M_final = 1

    #self.attention = attention(M,d_h,d_k,d_b)

    self.final = out_attention(d_h, d_h)
  
  def forward(self, h_c, H):
    return self.final(h_c, H)

class decoder(nn.Module):
  def __init__(self, M, n, d_h, d_k, d_v):
    super().__init__()
    self.M = M
    self.n = n
    self.d_h = d_h
    self.d_k = d_k
    self.d_v = d_v

    self.decoder_unit = decoder_unit(d_h)

    self.v_1 = nn.Parameter(torch.zeros(d_h))
    self.v_f = nn.Parameter(torch.zeros(d_h))

  ## missing mask
  def forward(self,n, H):
    B = H.shape[0]
    H_bar = 1/n * torch.sum(H, dim=-2)
    H_bar = H_bar.view(B,1,-1)
    h_c = torch.cat((H_bar, self.v_1.repeat(B,1).view(B,1,-1), self.v_f.repeat(B,1).view(B,1,-1)),dim=-1)

    for i in range(n):
      h_c.view(B,1,-1)
      batch_probs = self.decoder_unit(h_c, H)
      Pi_i = torch.multinomial(batch_probs,1).view(B,1)
      if i == 0:
        Pi = Pi_i
      Pi = torch.cat((Pi, Pi_i), dim = -1)
      if (i == 0):
        Pi_1 = Pi_i

      #print(Pi_i.squeeze().shape)
      H_pi_i=H[range(B),Pi_i.squeeze(),:].view(B,1,-1)
      H_pi_1=H[range(B),Pi_1.squeeze(),:].view(B,1,-1)

      #print(H_pi_1.shape)

      h_c = torch.cat((H_bar, H_pi_i, H_pi_1), dim = -1)
      
      #the above code does the same thing as this for loop
      #this is the original code, but it was vectorized for speed
      """
      for b in range(B):
        h_bar = H_bar[b,:].view(1, 1, self.d_h)
        h_pi_i = H[b,Pi_i[b],:].view(1, 1, self.d_h)
        h_pi_1 = H[b,Pi_1[b],:].view(1, 1, self.d_h)

        h_c[b,:] = torch.cat((h_bar, h_pi_i, h_pi_1), dim=-1)
      """

    return Pi

class TSP_attention_net(nn.Module):
  def __init__(self, T, M, n, d_x, d_h, d_k, d_v, num_hidden_units, share_params=False, batch_norm=True):
    super().__init__()
    self.T = T
    self.M = M
    self.n = n
    self.d_h = d_h
    self.d_k = d_k
    self.d_v = d_v
    self.num_hidden_units = num_hidden_units
    self.share_params = share_params
    self.batch_norm = batch_norm

    self.encoder = encoder(T, M, n, d_x, d_h, d_k, d_v, num_hidden_units, share_params, batch_norm)

    self.decoder = decoder(M, n, d_h, d_k, d_v)

    self.encoding = None

  def forward(self, X):
    encoding = self.encoder(X)
    n = X.shape[-2]
    return self.decoder(n, encoding)


def run(model,loops,B,n,d_x):
  with torch.no_grad():
    for i in tqdm(range(loops)):
      X = torch.rand(size=(B,n,d_x))
      X = X.to(device)
      if(i % (loops//4) == 0):
        print(memory_usage())
      
      model.zero_grad()
      torch.cuda.empty_cache()
      encoding = model(X)
      #print(list(encoding))
    # print(len(encoding))

def main():

  ##### PARAMS ######
  d_x = 2
  d_h = 128
  M = 8
  T = 3
  n = 100
  d_k = d_q = d_h//M
  d_v = d_h//M
  num_hidden_units = 512
  batch_size = 512 #number of graphs to run simultaniously
  B = batch_size
  #num_channels = 1 #trying to determine if this is a good idea

  num_attention_params = T * M * 2 * d_h * (d_k + d_v)
  num_linear_params = T * 2 * (num_hidden_units * d_h) + d_h + num_hidden_units
  num_bn_params = T * 2 * 2 * d_h #I think?
  num_param = num_attention_params + num_linear_params + num_bn_params
  print(num_param)
  #######

  net = TSP_attention_net(T, M, n, d_x, d_h, d_k, d_v, num_hidden_units, batch_norm=False).to(device)

  run(net,2500//60,B,n,d_x)


main()