import decap_gen
import numpy as np


def plot_gen(probe, pi, model):
    n = 10
    m = 10
    with open("DPP_data/10x10_pkg_chip.npy", "rb") as f:
        raw_pdn = np.load(f)

    pi = pi.astype(int)
    z_final = decap_gen.decap_placement(n, m, raw_pdn, pi, probe)
    z_final = np.abs(z_final)

    return z_final


"""
def plot(n, m, freq_pts, probe, pi, model):
    with open ('15x15_pkg_chip.npy', 'rb') as f:
        raw_pdn=np.load(f)

    z_initial=decap_gen.initial_impedance(n,m,raw_pdn,probe)

    z_initial=np.abs(z_initial)


    with open ('plot_data/test10_initial.npy', 'wb') as f:
        np.save(f, z_initial)


    pi=pi.astype(int)

    z_final=decap_gen.decap_placement(n,m, raw_pdn,pi,probe, freq_pts)
    z_final=np.abs(z_final)


    if model==1:
        reward=model_1(freq_pts, z_initial, z_final)

    elif model==2:
        reward=model_2(freq_pts, z_initial, z_final)

    elif model==3:
        reward=model_3(freq_pts, z_initial, z_final)

    elif model==4:
        reward=model_4(freq_pts, z_initial, z_final)

    elif model==5:
        reward=model_5(freq_pts, z_initial, z_final)

    elif model==6:
        reward=model_6(freq_pts, z_initial, z_final)

    elif model==7:
        reward=model_7(freq_pts, z_initial, z_final)


    with open ('freq_201.npy', 'rb') as f:
        freq=np.load(f)

    plt.figure(1)
    plt.plot(freq, abs(z_final), 'black')
    plt.plot(freq, abs(z_initial), 'blue', '--')
    #plt.plot(freq, target_impedance, 'red')
    plt.yscale("log")
    plt.xscale("log")
    plt.show()
    print(reward)
"""
