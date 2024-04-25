import time

import numpy as np


# import skrf as rf
# from skrf import Network, Frequency
from numpy.linalg import inv

"""
num_decap=20
freq_pts=201

decap_snp=rf.Network('1nF_decap.s1p')
decap=rf.network.s2z(decap_snp._s, z0=50)

with open('1nF_decap.npy','wb') as f1:
    np.save(f1, decap)
assert(False)
with open('01nF_decap.npy', 'rb') as f: 
   decap = np.load(f)

decap=decap.reshape(-1)
z2=np.zeros((freq_pts, num_decap, num_decap))

for i in range (num_decap): 
    z2[:,i,i]=decap
      
with open ('z2_%d_decap.npy'%(num_decap), 'wb') as f:
     np.save(f, z2)
     
print(z2)
assert(False)

"""
# input: flattened decap_map of size 1, mxn


def decap_placement(n, m, raw_pdn, pi, probing_port, freq_pts, decap):
    num_decap = np.size(pi)
    probe = probing_port
    z1 = raw_pdn

    """
    with open('data/z2_%d_decap.npy'%num_decap, 'rb') as f: # NOTE: size(freq_pts, num_decap, num_decap) 
        z2 = np.load(f)
    """
    # with open("DPP_data/01nF_decap.npy", "rb") as f:
    #     decap = np.load(f)

    # decap = decap.reshape(-1)
    
    z2 = np.zeros((freq_pts, num_decap, num_decap))

    qIndx = []
    for i in range(num_decap):
        z2[:, i, i] = np.abs(decap)
        qIndx.append(i)
    pIndx = pi.astype(int)

    # pIndx : index of ports in z1 for connecting
    # qIndx : index of ports in z2 for connecting

    aIndx = np.arange(len(z1[0]))

    aIndx = np.delete(aIndx, pIndx)

    z1aa = z1[:, aIndx, :][:, :, aIndx]
    z1ap = z1[:, aIndx, :][:, :, pIndx]
    z1pa = z1[:, pIndx, :][:, :, aIndx]
    z1pp = z1[:, pIndx, :][:, :, pIndx]
    z2qq = z2[:, qIndx, :][:, :, qIndx]

    end = time.time()

    zout = z1aa - np.matmul(np.matmul(z1ap, inv(z1pp + z2qq)), z1pa)

    for i in range(n * m):
        if i in pi:

            if i < probing_port:
                probe = probe - 1

    probe = int(probe)
    zout = zout[:, probe, probe]
    return zout


def initial_impedance(n, m, raw_pdn, probe):
    probe = int(probe)
    zout = raw_pdn[:, probe, probe]

    return zout
