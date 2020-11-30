import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import corner
import copy
from AdvDiff_solver import AdvDiff_solve

def ln_likelihood(data_expmt, params):
    n = len(data_expmt)

    param_sig_Sq = sig_true**2

    diff = data_expmt-computational_model(beta_deterministic, bc_left_deterministic, bc_right_deterministic, params)

    return -0.5*n*np.log(2*np.pi*param_sig_Sq) - 0.5*np.sum(diff**2/param_sig_Sq)

def computational_model(beta, bc_left, bc_right, params):
    return (AdvDiff_solve(beta, params[1], params[0], bc_left, bc_right))


def ln_likelihood_poly_fit(data_experiment):
    Nsamples = 10
    samples_f = np.random.uniform(6.0, 14.0, Nsamples)
    samples_k = np.random.uniform(0.01,1.0, Nsamples)
    N_tot = Nsamples*Nsamples
    y = np.zeros((N_tot,1), dtype=float)
    A = np.zeros((N_tot,6), dtype=float)
    for i in range(N_tot):
        i2D = np.floor(i/Nsamples)
        j2D = i - Nsamples*i2D
        i2D = i2D.astype(int)
        j2D = j2D.astype(int)
        # print(i2D,' ',j2D)
        new_params = np.array([samples_f[i2D], samples_k[j2D]])
        y[i,0] = ln_likelihood(data_experiment, new_params)
        A[i,:] = [1, samples_f[i2D], samples_f[i2D]**2, np.log(samples_k[j2D]), samples_k[j2D], samples_f[i2D]*samples_k[j2D]]

    y = np.matmul(np.transpose(A), y)
    A = np.matmul(np.transpose(A), A)
    a = np.linalg.solve(A,y)

    return a

np.random.seed(12345)

n_data = 50

# true source
source_true = 10.0
kappa_true = 0.2
# true noise added to data
sig_true = 1.0

# deterministic parameters
beta_deterministic     = 2.0
bc_left_deterministic  = 0.5
bc_right_deterministic = 0.0

# clean data
data_true = AdvDiff_solve(beta_deterministic, kappa_true, source_true, bc_left_deterministic, bc_right_deterministic)
# random noise
noise_true = np.random.normal(0,sig_true,n_data)
# noisy data
data_experiment = data_true + noise_true


params0 = np.linspace(6.0,14.0, num=20)
params0 = np.reshape(params0, (20,1))
params1 = np.linspace(0.01,1.0, num=20)
params1 = np.reshape(params1, (20, 1))

new_ln_like = np.zeros((20,20), dtype=float)
for i in range(20):
    for j in range(20):
        new_params = [params0[i,0], params1[j,0]]
        new_ln_like[i,j] = ln_likelihood(data_experiment, new_params)


a_coeffs = ln_likelihood_poly_fit(data_experiment)

new_approx = np.zeros((20,20), dtype=float)
for i in range(20):
    f_cont = a_coeffs[0] + a_coeffs[1]*params0[i,0] + a_coeffs[2]*params0[i,0]**2
    for j in range(20):
        k_cont = a_coeffs[3]*np.log(params1[j,0]) + a_coeffs[4]*params1[j,0] + a_coeffs[5]*params1[j,0]*params0[i,0]
        new_approx[i,j] = f_cont + k_cont

xf, xk = np.meshgrid(params0,params1)

fig = plt.figure()
ax1 = fig.add_subplot(121,projection='3d')

# Plot the surface.
surf1 = ax1.plot_surface(xf, xk, np.transpose(new_ln_like), cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

fig.colorbar(surf1, shrink=0.5, aspect=5)


ax2 = fig.add_subplot(122,projection='3d')

# Plot the surface.
surf2 = ax2.plot_surface(xk, xf, np.transpose(new_approx), cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

fig.colorbar(surf2, shrink=0.5, aspect=5)

# ax3 = fig.add_subplot(122,projection='3d')
# surf3 = ax3.plot_surface(xf, xk, new_approx-new_ln_like, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
#
# fig.colorbar(surf3, shrink=0.5, aspect=5)

plt.show()

# plt.plot(params0,np.exp(new_ln_like))
# plt.plot(params0,new_ln_like)
# plt.figure()
# plt.plot(params0,new_approx,'r--',params0,new_ln_like,'bo')
# plt.show()
#
# plt.figure()
# plt.plot(params0,np.exp(new_approx),'r--',params0,np.exp(new_ln_like),'bo')
# plt.show()
