import numpy as np
import matplotlib.pyplot as plt
import corner
import copy
from AdvDiff_solver import AdvDiff_solve

np.random.seed(12345)

n_data = 50

# true source
source_true = 10.0
# true noise added to data
sig_true = 1.0

# deterministic parameters
kappa_deterministic    = 0.1
beta_deterministic     = 2.0
bc_left_deterministic  = 0.5
bc_right_deterministic = 0.0

# clean data
data_true = AdvDiff_solve(beta_deterministic, kappa_deterministic, source_true, bc_left_deterministic, bc_right_deterministic)
# random noise
noise_true = np.random.normal(0,sig_true,n_data)
# noisy data
data_experiment = data_true + noise_true

def ln_prior(params):
    # source : [-10, 10]
    # sigma : [0, 10]
    nrv = len(params0)
    params = np.reshape(params,[1,nrv])
    # print(params)
    source  = params[0,0]
    ln_prior_val = 0

    if source < 5 or source > 20:
        return -np.inf
    else:
        ln_prior_val += np.log(1.0/15.0)

    if (len(params)==2):
        sig = params[0,1]
        if sig < 0 or sig > 10:
            return -np.inf
        else:
            ln_prior_val += np.log(1.0/10.0)

    return ln_prior_val

def computational_model(beta, kappa, bc_left, bc_right, params):
    return (AdvDiff_solve(beta, kappa, params[0,0], bc_left, bc_right))

def ln_likelihood(data_expmt, params):
    nrv = len(params0)
    params = np.reshape(params,[1,nrv])
    n = len(data_expmt)

    param_sig_Sq = sig_true**2
    if (nrv==2):
        param_sig_Sq = params[0,1]**2

    diff = data_expmt-computational_model(beta_deterministic, kappa_deterministic, bc_left_deterministic, bc_right_deterministic, params)

    return -0.5*n*np.log(2*np.pi*param_sig_Sq) - 0.5*np.sum(diff**2/param_sig_Sq)

def ln_posterior(data_expmt, params):
    ln_prior_val = ln_prior(params)
    if (np.isinf(ln_prior_val)): # skip likelihood computation
        return ln_prior_val

    return ln_likelihood(data_expmt, params)+ln_prior_val

def run_metropolis_hastings(params0, n_steps, proposal_sigmas):
    print("Running MCMC-MH...\n")
    if (len(proposal_sigmas) != len(params0)):
        raise ValueError("Proposal distribution should have same shape as parameter vector.")

    chain = np.zeros((n_steps, len(params0)))
    ln_posts = np.zeros(n_steps)

    n_accept = 0

    chain[0] = params0
    ln_posts[0] = ln_posterior(data_experiment, chain[0])

    # loop through the number of steps requested and run MCMC
    for i in range(1,n_steps):
        print("\nIteration--",i)
        # proposed new parameters
        new_params = np.random.normal(chain[i-1],proposal_sigmas)

        new_ln_post = ln_posterior(data_experiment, new_params)

        ln_p_accept = new_ln_post - ln_posts[i-1]

        ln_r = np.log(np.random.rand())

        if (ln_p_accept > ln_r):
            chain[i] = new_params
            ln_posts[i] = new_ln_post
            n_accept += 1
        else:
            chain[i] = chain[i-1]
            ln_posts[i] = ln_posts[i-1]

    acc_frac = n_accept / n_steps
    return chain, ln_posts, acc_frac


def run_dram(params0, n_steps, init_cov, n_AM, n_up, gamma_DR):

    if (init_cov.shape[0] != init_cov.shape[1] or init_cov.shape[0] != len(params0)):
        raise ValueError("Proposal covariance should have same shape as parameter vector.")

    s_AM = 2.38**2/len(params0)
    n_burn = 800

    cov = copy.deepcopy(init_cov)
    L = np.linalg.cholesky(cov)

    chain = np.zeros((n_steps, len(params0)))
    ln_posts = np.zeros(n_steps)

    n_accept = 0

    chain[0] = params0
    ln_posts[0] = ln_posterior(data_experiment, chain[0])

    # loop through the number of steps requested and run MCMC
    for i in range(1,n_steps):
        print("\nIteration--",i)
        # print(chain[i-1])
        # proposed new parameters
        z = np.random.normal(0,np.ones(len(params0)))
        new_params = chain[i-1]+L@z

        new_ln_post = ln_posterior(data_experiment, new_params)

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

            new_ln_post2 = ln_posterior(data_experiment, new_params2)

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
            L = np.linalg.cholesky([[cov]])

    acc_frac = n_accept / n_steps
    return chain, ln_posts, acc_frac


# case 1: sig_true is known
run_case1 = False
if (run_case1):

    # params0 = [15.0]
    # proposal_sigmas = [0.25]
    # n_steps = 4096
    #
    # chain,_,acc_frac = run_metropolis_hastings(params0, n_steps, proposal_sigmas)

    params0 = [15.0]
    init_cov = np.diag([(0.25)**2])
    n_steps = 2048
    n_AM = 512
    n_up = 256
    gamma_DR = 1./5.

    chain,_,acc_frac = run_dram(params0, n_steps, init_cov, n_AM, n_up, gamma_DR)

    print('case 1 stats:')
    print('  acceptance fraction: {:.2%}'.format(acc_frac))

    good_samples = chain[1024::4] # discard first 1024 samples and take every 4th

    low,med,hi = np.percentile(good_samples, [16, 50, 84], axis=0)
    upper, lower = hi-med, med-low

    for i,name in enumerate(['source:']):
        print(' ',name,' %.4f'%med[i],'+/- %.4f'%upper[i],'/%.4f'%lower[i])

    plot_each_chain = True
    if(plot_each_chain):
        plt.plot(chain[:,0], color='k', drawstyle='steps')
        plt.axhline(source_true, color='r', label='true')
        plt.ylabel('$f$/source')

        plt.savefig('MCMC_DRAM_f_chains.png')
        plt.show()

    plot_margPDFs_using_corner = True
    if (plot_margPDFs_using_corner):
        # fig = corner.corner(chain[1024:], bins=32, labels=['$beta$/advection'], truths=[beta_true])
        plt.hist(chain[1024:], bins=32, range=(9.3,10.7))
        plt.axvline(source_true, color='r', label='true')
        plt.savefig('MCMC_DRAM_f_margPDFs.png')
        plt.show()

# case 2: sig_true is unknown
run_case2 = True
if (run_case2):

    params0 = [6.0, 3.0]
    init_cov = np.diag([0.05**2,0.05**2])
    n_steps = 2048
    n_AM = 512
    n_up = 256
    gamma_DR = 1./5.

    chain,_,acc_frac = run_dram(params0, n_steps, init_cov, n_AM, n_up, gamma_DR)

    print('case 1 stats:')
    print('  acceptance fraction: {:.2%}'.format(acc_frac))

    good_samples = chain[1024::4] # discard first 1024 samples and take every 4th

    low,med,hi = np.percentile(good_samples, [16, 50, 84], axis=0)
    upper, lower = hi-med, med-low

    for i,name in enumerate(['source:','sigma']):
        print(' ',name,' %.4f'%med[i],'+/- %.4f'%upper[i],'/%.4f'%lower[i])

    plot_chain_in_paramSpace_case1 = True
    if (plot_chain_in_paramSpace_case1):
        plt.plot(source_true, sig_true, marker='o', color='r', zorder=10)
        plt.plot(chain[:,0], chain[:,1], marker='', color='k', linewidth=1.)
        plt.xlabel('source')
        plt.ylabel('$\sigma$')

        plt.savefig('MCMC_MH_f-sig_paramspace.png')
        plt.show()

    plot_each_chain = True
    if(plot_each_chain):
        fig,axes = plt.subplots(len(params0), 1, sharex=True)

        for i in range(len(params0)):
            axes[i].plot(chain[:,i], color='k', drawstyle='steps')

        axes[0].axhline(source_true, color='r', label='source_true')
        axes[0].legend(loc='best')
        axes[0].set_ylabel('$f$/source')

        axes[1].axhline(sig_true, color='r', label='sigma_true')
        axes[0].legend(loc='best')
        axes[1].set_ylabel('$\sigma_{\epsilon}$/noise')

        plt.savefig('MCMC_MH_f-sig_chains.png')
        plt.show()


    # plot_margPDFs_using_corner = True
    # if (plot_margPDFs_using_corner):
    #     # fig = corner.corner(chain[1024:], bins=32, labels=['$beta$/advection'], truths=[beta_true])
    #     plt.hist(chain[1024:], bins=32, range=(9.3,10.7))
    #     plt.axvline(source_true, color='r', label='true')
    #     plt.savefig('MCMC_DRAM_case0_margPDFs_1rv.png')
    #     plt.show()

    plot_margPDFs_using_corner = True
    if (plot_margPDFs_using_corner):
        fig = corner.corner(chain[1024:], bins=32, labels=['$f$/source', '$\sigma_{\epsilon}$/noise'], truths=[source_true, sig_true])

        plt.savefig('MCMC_MH_f-sig_margPDFs.png')
        plt.show()
