import os

import decap_gen
import numpy as np

def model_1(freq_pts, z_initial, z_final, freq):
    impedance_gap = np.zeros(freq_pts)

    freq_point = 2e9
    min = 0.32
    grad = 0.16
    target_impedance = np.zeros(np.shape(freq))
    idx0 = np.argwhere(freq < freq_point)
    idx1 = np.argwhere(freq >= freq_point)
    target_impedance[idx0] = min
    target_impedance[idx1] = grad * 1e-9 * freq[idx1]

    penalty = 1
    reward = 0

    for i in range(freq_pts):
        if z_final[i] > target_impedance[i]:
            impedance_gap[i] = (z_final[i] - target_impedance[i]) * penalty
        else:
            impedance_gap[i] = 0
        # impedance_gap[i]=target_impedance[i]-z_final[i]

        reward = reward - (impedance_gap[i] / (434 * penalty))
    return reward


def model_2(freq_pts, z_initial, z_final, freq):

    impedance_gap = np.zeros(freq_pts)

    reward = 0

    for i in range(freq_pts):
        impedance_gap[i] = z_initial[i] - z_final[i]
        reward = reward + impedance_gap[i]
    reward = reward / 10
    return reward


def model_3(freq_pts, z_initial, z_final, freq):

    impedance_gap = np.zeros(freq_pts)

    freq_point = 2e9
    reward = 0

    for i in range(freq_pts):
        impedance_gap[i] = z_initial[i] - z_final[i]

        if freq[i] < freq_point:
            reward = reward + (impedance_gap[i] * 1.5)

        else:
            reward = reward + impedance_gap[i]
    reward = reward / 10
    return reward


def model_4(freq_pts, z_initial, z_final, freq):

    impedance_gap = np.zeros(freq_pts)

    freq_point = 2e9
    reward = 0

    for i in range(freq_pts):
        impedance_gap[i] = z_initial[i] - z_final[i]

        if freq[i] < freq_point:
            if impedance_gap[i] > 0:
                reward = reward + (impedance_gap[i] * 1.5)
            else:
                reward = reward + (impedance_gap[i] * 3)
        else:
            if impedance_gap[i] > 0:
                reward = reward + impedance_gap[i]
            else:
                reward = reward + (impedance_gap[i] * 3)
    reward = reward / 10
    return reward


def model_5(freq_pts, z_initial, z_final, freq):

    impedance_gap = np.zeros(freq_pts)

    # reward = 0
    # for i in range(freq_pts):
    #     impedance_gap[i] = z_initial[i] - z_final[i]
    #     reward = reward + (impedance_gap[i] * 1000000000 / freq[i])
    # reward = reward / 10
    # return reward

    # vectorized version
    impedance_gap = z_initial - z_final
    reward = np.sum(impedance_gap * 1000000000 / freq) / 10
    return reward


def model_6(freq_pts, z_initial, z_final, freq=None):

    impedance_gap = np.zeros(freq_pts)
    target_impedance = 0.6 * np.ones(freq_pts)
    reward = 0
    penalty = 1

    for i in range(freq_pts):  # NOTE: 0.013 sec
        if z_final[i] > target_impedance[i]:  # NOTE: size(434)
            impedance_gap[i] = (z_final[i] - target_impedance[i]) * penalty
        else:
            impedance_gap[i] = 0
            # impedance_gap[i]=target_impedance[i]-z_final[i]
        reward = reward - (
            impedance_gap[i] / (434 * penalty)
        )  # TODO: Using torch.mean()

    return reward


def model_7(freq_pts, z_initial, z_final, freq=None):

    impedance_gap = np.zeros(freq_pts)
    target_impedance = np.ones(freq_pts)
    reward = 0
    penalty = 1

    for i in range(freq_pts):  # NOTE: 0.013 sec
        if z_final[i] > target_impedance[i]:  # NOTE: size(434)
            impedance_gap[i] = (z_final[i] - target_impedance[i]) * penalty
        else:
            impedance_gap[i] = 0
            # impedance_gap[i]=target_impedance[i]-z_final[i]
        reward = reward - (
            impedance_gap[i] / (434 * penalty)
        )  # TODO: Using torch.mean()

    return reward


class RewardModel:
    def __init__(self, 
                    basepath,
                    model_number=5,
                    freq_pts = 201,
                    n=10,
                    m=10,
                    freq_data_path="DPP_data/freq_201.npy",
                    raw_pdn_path="DPP_data/10x10_pkg_chip.npy"):
        self.model_number = model_number
        self.freq_pts = freq_pts
        self.n = n
        self.m = m
        
        self.basepath = basepath
        
        freq_data_path = os.path.join(basepath, freq_data_path)
        raw_pdn_path = os.path.join(basepath, raw_pdn_path)
        self.freq = self.load_data(freq_data_path)
        self.raw_pdn = self.load_data(raw_pdn_path)
        
        decap_path = os.path.join(basepath, "DPP_data/01nF_decap.npy")
        with open(decap_path, "rb") as f:
            self.decap = np.load(f).reshape(-1)

        # get reward model based on model number       
        class_name = "model_" + str(model_number) # e.g. get model_5 as function
        self.model = globals()[class_name]
        
    def load_data(self, path):
        with open(path, "rb") as f:
            return np.load(f)
        
    def __call__(self, probe, pi):
        z_initial = decap_gen.initial_impedance(self.n, self.m, self.raw_pdn, probe)
        z_initial = np.abs(z_initial)

        pi = pi.astype(int)

        z_final = decap_gen.decap_placement(self.n, self.m, self.raw_pdn, pi, probe, self.freq_pts, self.decap)
        z_final = np.abs(z_final)
        
        return self.model(self.freq_pts, z_initial, z_final, self.freq)
        
        
if __name__ == "__main__":
    
    n, m = 10, 10
    reward_model = RewardModel(n=n, m=m)
    
    # choose 11 numbers between 0 and n*m (unique)
    rand_num = np.random.choice(n*m, 11, replace=False)
    probe = rand_num[0]
    pi = rand_num[1:]
    reward = reward_model(probe, pi)
    print(reward)            
    

