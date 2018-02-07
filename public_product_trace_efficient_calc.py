# -*- coding: utf-8 -*-
"""
Code for efficient calculation of the trace of the kernel product for measuring multimodal correspondence 
The code includes validation of the calculation by comparison to batch calculation

If you use this code please cite:
D. Dov, I. Cohen, and R. Talmon, “Sequential audio-visual correspondence with alternating diffusion kernels,” IEEE Transactions on Signal Processing, vol. PP, no. 99, pp. 1–1, 2018.

Date: 2.2018

Author: David Dov
"""
#%% imports
import numpy as np
from scipy.spatial.distance import cdist     
from numpy.random import rand


#%% alternating diffusion maps - the product of affinity kernels
def adm(feat1, feat2, sig1, sig2):    
    # calculate the kernel product
    # inputs:  feat1, feat2 - features of modalities 1 and 2    
    #          sig1, sig2 - kernel bandwidth of modalities 1 and 2    
    #
    # outputs: M - the kernel product
    #          W1, W2 - affinity kernels of modalities 1 and 2  
    #          D1, D2 - normalization (degree) matrices of modalities 1 and 2  
    
    M1, W1, D1 = calc_norm_ker(feat1, sig1)  # calculate normalized affinity kernel in modality 1
    M_batch, W2, D2 = calc_norm_ker(feat2, sig2)  # calculate normalized affinity kernel in modality 2
    M = M1.dot (M_batch.T) # kernel product
    return M, W1, D1, W2, D2
    
    
def calc_norm_ker(feat, sig): #calculate row normalized kernel for a single modality     
    dist = cdist(feat.T,feat.T,'euclidean')                  
    W = np.exp(-(dist**2 / sig))
    D = np.sum(W, axis = 0) #
    invD = np.diag(D**-1)
    M = invD.dot(W)  # row normalization
    return M, W, D


#%% update kernels
def updateKer(W_prev, W_N1):    
    # update the affinity kernel of a single modality by removing the first row and column and updating the new ones
    # inputs: W_prev - prvious kernel
    #        W_N1: affinities to the new frame N+1
    # output: W - updated kernel
    W = np.zeros(W_prev.shape)
    W[:-1, :-1] = W_prev[1:, 1:]
    W[-1,:] = W_N1
    W[:, -1] = W_N1
    return W

#%% update W and D
def updateWD(W_prev, D1_prev, D2_prev, W1_prev, W2_prev, W1_N1, W2_N1):      
    D1 = np.zeros(D1_prev.shape)       
    D1[:-1] = D1_prev[1:] - W1_prev[0, 1:] + W1_N1[:-1]
    D1[-1] = np.sum(W1_N1)
    
    D2 = np.zeros(D2_prev.shape)
    D2[:-1] = D2_prev[1:] - W2_prev[0, 1:] + W2_N1[:-1]
    D2[-1] = np.sum(W2_N1)
    
    D = (D1 * D2) ** -1                         
    
    W = np.zeros(W_prev.shape)        
    W[:-1] = W_prev[1:] - W1_prev[0, 1:] * W2_prev[0, 1:] + W1_N1[:-1]*W2_N1[:-1]
    W[-1] = np.sum(W1_N1*W2_N1)
    return W, D, D1, D2

#%% calculate affinities to the new frame
def calcAff(sig, feat):              
    # calculate the affinity between the new frame N+1 to frames 2,3,...,N
    # inputs: feat - features of frames 2,3,...,N+1
    #         sig - the kernel bandwidth
    # output: W_N1 - affinities to frame N+1
    
    featN1 = feat[:, -1]
    featN1 = np.expand_dims(featN1, axis=1)
    dist = cdist(featN1.T,feat.T,'euclidean')        
    W_N1 = np.exp(-(dist**2/sig))    
    W_N1 = np.squeeze(W_N1)
    return W_N1# the N+1 (i.e., the new incoming) row/column of the affinity kernel
    
#%% initialize vectors W and D
def initiateWD(W1, D1, W2, D2):    
    D = (D1 * D2) ** -1
    W = np.sum(W1 * W2, axis=0)                
    return W, D

#%% main 
N = 100 # batch size
feat1 = rand(13, N + 1) # features of modality 1
feat2 = rand(20, N + 1) # features of modality 2
sig1 = 1; sig2 = 1 # kernel bandwidths

# 
# initialization, calculate the kernel product for the previous frame, i.e., franes 1,2,...,N
M_prev, W1_prev, D1_prev, W2_prev, D2_prev = adm(feat1[:, :-1], feat2[:, :-1], sig1, sig2)            

# initialize vectors W and D
W_prev, D_prev = initiateWD(W1_prev, D1_prev, W2_prev, D2_prev)

# for validation, calculate the kernel product in a batch manner for the new frame, i.e., franes 2,2,...,N+1 
M_batch, _, _, _, _= adm(feat1[:, 1:], feat2[:, 1:], sig1, sig2)            

# calculate affinities to the new frames
W1_N1 = calcAff(sig1, feat1[:, 1:]) 
W2_N1 = calcAff(sig2, feat2[:, 1:]) 

# update W and D
W, D, D1, D2 = updateWD(W_prev, D1_prev, D2_prev, W1_prev, W2_prev, W1_N1, W2_N1)

# the trace of the product of kernels
Mtrace = np.sum(W * D)

#validate by comparing to the batch approach
print 'validate trace: ' + str(Mtrace - M_batch.trace())

# for the next iteration, update affinity kernels
W1 = updateKer(W1_prev, W1_N1)
W2 = updateKer(W2_prev, W2_N1)

