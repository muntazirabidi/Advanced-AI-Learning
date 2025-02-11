import os
import numpy as np
import camb

def generate_camb_cl(params):
    """Generate CMB power spectrum using CAMB"""
    H0, ombh2, omch2, logA, ns = params
    As = 1e-10 * np.exp(logA)
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=0.06, tau=0.06)
    pars.InitPower.set_params(As=As, ns=ns, r=0)
    pars.set_for_lmax(2500, lens_potential_accuracy=0)
    results = camb.get_results(pars)
    return results.get_cmb_power_spectra(pars)['total'][2:, 0]

def generate_sims(num_samples, param_ranges):
    """Generate simulation dataset"""
    params = np.random.uniform(
        low=param_ranges[:, 0], 
        high=param_ranges[:, 1],
        size=(num_samples, 5)
    )
    
    cls = np.array([generate_camb_cl(p) for p in params])
    return params, cls