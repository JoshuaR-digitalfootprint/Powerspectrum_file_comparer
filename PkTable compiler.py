# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 14:39:54 2026

@author: joshu
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad

import matplotlib.pyplot as plt


import glob


def get_data(files):

    num_files = len(files)
    print(f"Found {num_files} .dat files in the directory")

    # get the number of k values from the first file
    first_file = files[0]
    k = np.loadtxt(first_file, comments="#", usecols=0 )
    num_k = len(k)
    print(f"Number of k values in the first file: {num_k}")

    # initialise empty array to hold all P(k) values
    all_Pk = np.zeros((num_files, num_k))

    # loop through each file and read the P(k) values
    for i, file in enumerate(files):
        Pk = np.loadtxt(file, comments="#", usecols=1 )
        all_Pk[i, :] = Pk

    return k, all_Pk


def top_hat_window(x):
    """
    Fourier transform of a 3D spherical top-hat window function W(x).
    Uses a Taylor series expansion for x < 1e-3 to prevent 0/0 numerical errors.

    Parameters:
        x: flt
    Returns:
        w: flt
    """
    x = np.asarray(x, dtype=float)

    # Pre-allocate array
    w = np.empty_like(x)

    # Mask for tiny values where x -> 0 limit is needed
    small = np.abs(x) < 1e-3
    large = ~small # large is not small

    # Taylor expansion: 1 - x^2/10 + x^4/280 for approximation
    if np.any(small):
        xs = x[small]
        w[small] = 1.0 - (xs**2) / 10.0 + (xs**4) / 280.0

    # Standard: 3 * (sin(x) - x*cos(x)) / x^3
    if np.any(large):
        xl = x[large]
        w[large] = 3.0 * (np.sin(xl) - xl * np.cos(xl)) / (xl**3)

    return w


def integrand(k, pk_func, R=8.):
    x = k * R
    w = top_hat_window(x)
    return (1.0 / (2.0 * np.pi**2)) * (k**2) * pk_func(k) * (w**2)


def compute_sigma8(k, pk, R=8.0, plot=False):
    """
    Computes sigma_R using log-space integration.
    """


    # Plot
    if plot:
        plt.figure(figsize=(10, 6))
        plt.loglog(k, pk, '.', label='Data points', markersize=8)
        #plt.plot(log_k_smooth, log_pk_smooth, '-', label='Cubic interpolation', linewidth=2)
        plt.xlabel('log(k)')
        plt.ylabel('log(P(k))')
        plt.title('Power Spectrum with Cubic Interpolation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    k_min = np.min(k)
    k_max = np.max(k)

    pk_func = interp1d(k, pk, kind='linear')

    # quad automatically uses adaptive quadrature for accuracy
    sigma2, _ = quad(integrand, k_min, k_max, limit=500, args=(pk_func, R))
    
    
    
    

    return np.sqrt(sigma2)

# def Om_Lambda(files):
#     """
#     return:
#         all_value
#     """
#     num_files = len(files)
#     first_file = files[0]
#     k = np.loadtxt(first_file, usecols=0)
#     num_k = len(k)
#     all_value = np.zeros((num_files, num_k))
#     for i, file in enumerate(files):
#         with open(file, 'r') as f:
#             lines = f.readlines()
#             value = float(lines[2].split('=')[1].strip())
#             all_value[i, :] = value

#     return all_value




def main():
    files = glob.glob(r'C:\Users\joshu\Documents\summer 2026 n-body\pk_pool\pk_pool\*.dat')

    k, all_Pk = get_data(files)
    
    #value = Om_Lambda(files)
    
    # Compute sigma8 for each dataset
    sigma8_values = []
    for i in range(all_Pk.shape[0]):  # Limit to first two files for demonstration
        print(f"Processing file: {files[i]}")
        sigma8 = compute_sigma8(k, all_Pk[i, :], R=8.0)
        sigma8_values.append(sigma8)
        
        print(f"File: {files[i]}, Sigma_8: {sigma8}")
        

    sigma8 = np.array(sigma8_values)
    np.savetxt(r'all_sigma8.txt', sigma8)
    



if __name__ == "__main__":
    main()