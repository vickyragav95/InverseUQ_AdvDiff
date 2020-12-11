import numpy as np
import matplotlib.pyplot as plt
import corner
import copy
from AdvDiff_solver import AdvDiff_solve

np.random.seed(12345)

n_data = 50

# true diffusivity
kappa_true = 0.2
# true source
source_true = 10.0
# true noise added to data
sig_true = 3.0

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

def ln_prior(params):
    # kappa : [0.0, 0.5]
    # source : [5, 20]
    # sigma : [0, 10]
    kappa  = params[1]
    source = params[0]
    ln_prior_val = 0

    if kappa < 0.0 or kappa > 2.0:
        return -np.inf
    else:
        ln_prior_val += np.log(1.0/2.0)

    if source < 5 or source > 20:
        return -np.inf
    else:
        ln_prior_val += np.log(1.0/15.0)

    return ln_prior_val

def computational_model(beta, bc_left, bc_right, params):
  return (AdvDiff_solve(beta, params[0,1], params[0,0], bc_left, bc_right))

def ln_likelihood(data_expmt, params):
    n = len(data_expmt)
    nrv = len(params0)
    params = np.reshape(params,[1,nrv])

    param_sig_Sq = sig_true**2

    diff = data_expmt-computational_model(beta_deterministic, bc_left_deterministic, bc_right_deterministic, params)

    return -0.5*n*np.log(2*np.pi*param_sig_Sq) - 0.5*np.sum(diff**2/param_sig_Sq)

def ln_likelihood_simple(data_expmt, params):
    nrv = len(params0)
    params = np.reshape(params,[1,nrv])

    diff = data_expmt-computational_model(beta_deterministic, bc_left_deterministic, bc_right_deterministic, params)

    return -0.5*np.sum(diff**2)

def ln_likelihood_construct_surrogate(data_experiment):
    Nsamples_f = 10
    Nsamples_k = 20
    samples_f = np.random.uniform(8.0, 12.0, Nsamples_f)
    samples_k = np.random.uniform(0.01,0.5, Nsamples_k)
    N_tot = Nsamples_f*Nsamples_k
    y = np.zeros((N_tot,1), dtype=float)
    A = np.zeros((N_tot,6), dtype=float)
    for i in range(N_tot):
        j2D = np.floor(i/Nsamples_f)
        i2D = i - Nsamples_f*j2D
        i2D = i2D.astype(int)
        j2D = j2D.astype(int)
        # print(i2D,' ',j2D)
        new_params = np.array([samples_f[i2D], samples_k[j2D]])
        y[i,0] = ln_likelihood_simple(data_experiment, new_params)
        A[i,:] = [1, samples_f[i2D], samples_f[i2D]**2, np.log(samples_k[j2D]), samples_k[j2D], samples_f[i2D]*samples_k[j2D]]

    y = np.matmul(np.transpose(A), y)
    A = np.matmul(np.transpose(A), A)
    a = np.linalg.solve(A,y)

    return a

def surrogate_ln_likelihood(a_like, params):
    nrv = len(params0)
    params = np.reshape(params,[1,nrv])

    kappa = params[0,1]
    source = params[0,0]
    param_sig_Sq = sig_true**2

    fk_model = a_like[0] + a_like[1]*source + a_like[2]*source**2 + a_like[3]*np.log(kappa) + a_like[4]*kappa + a_like[5]*kappa*source
    return fk_model/param_sig_Sq

def ln_posterior(data_expmt, params, surr_model):
    ln_prior_val = ln_prior(params)
    if (np.isinf(ln_prior_val)): # skip likelihood computation
        return ln_prior_val

    if (surr_model):
        Llk = surrogate_ln_likelihood(a_like, params)
    else:
        Llk = ln_likelihood(data_expmt, params)

    return Llk+ln_prior_val

def run_dram(params0, n_steps, init_cov, n_AM, n_up, gamma_DR, surr_model):

    if (init_cov.shape[0] != init_cov.shape[1] or init_cov.shape[0] != len(params0)):
        raise ValueError("Proposal covariance should have same shape as parameter vector.")

    s_AM = 2.38**2/len(params0)
    n_burn = 1024

    cov = copy.deepcopy(init_cov)
    L = np.linalg.cholesky(cov)

    chain = np.zeros((n_steps, len(params0)))
    ln_posts = np.zeros(n_steps)

    n_accept = 0

    chain[0] = params0
    ln_posts[0] = ln_posterior(data_experiment, chain[0], surr_model)

    # loop through the number of steps requested and run MCMC
    for i in range(1,n_steps):
        print("\nIteration--",i)
        # proposed new parameters
        z = np.random.normal(0,np.ones(len(params0)))
        new_params = chain[i-1]+L@z

        new_ln_post = ln_posterior(data_experiment, new_params, surr_model)

        ln_p_accept = new_ln_post - ln_posts[i-1]

        ln_r = np.log(np.random.rand())

        if (ln_p_accept > ln_r):
            chain[i] = new_params
            ln_posts[i] = new_ln_post
            n_accept += 1
        else:
            # proposed new parameters
            z = np.random.normal(0,np.ones(len(params0)))
            new_params2 = chain[i-1]+gamma_DR*L@z

            new_ln_post2 = ln_posterior(data_experiment, new_params2, surr_model)

            ln_p_accept2 = new_ln_post - ln_posts[i-1]

            ln_r2 = np.log(np.random.rand())

            if (ln_p_accept2 > ln_r2):
                chain[i] = new_params2
                ln_posts[i] = new_ln_post2
                n_accept += 1
            else :
                chain[i] = chain[i-1]
                ln_posts[i] = ln_posts[i-1]

        if (i>n_burn and (i-n_burn)%n_up==0):
            cov = s_AM*np.cov(chain[max(n_burn,i-n_AM):i,:],rowvar=False)
            L = np.linalg.cholesky(cov)

    acc_frac = n_accept / n_steps
    return chain, ln_posts, acc_frac


# case 1: sig_true is known
run_case1 = True
if (run_case1):

    params0 = [15.0, 1.0]
    init_cov = np.diag([(0.25)**2, (0.02)**2])
    n_steps = 4096
    n_AM = 512
    n_up = 256
    gamma_DR = 1./5.

    surr_model = False
    a_like = np.zeros([6,1])
    if (surr_model):
        a_like = ln_likelihood_construct_surrogate(data_experiment)

    chain,_,acc_frac = run_dram(params0, n_steps, init_cov, n_AM, n_up, gamma_DR, surr_model)

    print('case 1 stats:')
    print('  acceptance fraction: {:.2%}'.format(acc_frac))

    good_samples = chain[1024::4] # discard first 1024 samples and take every 4th

    low,med,hi = np.percentile(good_samples, [16, 50, 84], axis=0)
    upper, lower = hi-med, med-low

    for i,name in enumerate(['source:','kappa:']):
        print(' ',name,' %.4f'%med[i],'+/- %.4f'%upper[i],'/%.4f'%lower[i])

    plot_chain_in_paramSpace_case1 = True
    if (plot_chain_in_paramSpace_case1):
        plt.plot(source_true, kappa_true, marker='o', color='r', zorder=10)
        plt.plot(chain[:,0], chain[:,1], marker='', color='k', linewidth=1.)

        plt.ylabel('$kappa$/diffusivity')
        plt.xlabel('$f$/source')
        if (surr_model):
            plt.savefig('MCMC_DRAM_k-f_paramspace_SM.png')
        else:
            plt.savefig('MCMC_DRAM_k-f_paramspace.png')
        plt.show()

    plot_each_chain = True
    if(plot_each_chain):
        fig,axes = plt.subplots(len(params0), 1, sharex=True)

        for i in range(len(params0)):
            axes[i].plot(chain[:,i], color='k', drawstyle='steps')

        axes[0].axhline(source_true, color='r', label='true')
        axes[0].legend(loc='best')
        axes[0].set_ylabel('$f$/source')

        axes[1].axhline(kappa_true, color='r', label='true')
        axes[1].legend(loc='best')
        axes[1].set_ylabel('$\kappa$/diffusivity')
        if (surr_model):
            plt.savefig('MCMC_DRAM_k-f_chains_SM.png')
        else:
            plt.savefig('MCMC_DRAM_k-f_chains.png')
        plt.show()

    plot_margPDFs_using_corner = True
    if (plot_margPDFs_using_corner):
        fig = corner.corner(chain[1024:], bins=32, labels=['$f$/source','$\kappa$/diffusivity'], truths=[source_true, kappa_true])
        if (surr_model):
            plt.savefig('MCMC_DRAM_k-f_margPDFs_SM.png')
        else:
            plt.savefig('MCMC_DRAM_k-f_margPDFs.png')
        plt.show()
