# profile_analysis.py - Likelihood/Objective function
# --------------------------------------------------
# This file is a part of DeerLab. License is MIT (see LICENSE.md).
# Copyright(c) 2019-2021: Luis Fabregas, Stefan Stoll and other contributors.

import numpy as np
from scipy.stats import chi2 
from deerlab import fit
from deerlab import noiselevel,UQResult
import warnings 
from tqdm import tqdm

def profile_analysis(model,y, *args, parameters='all', samples=50, noiselvl=None, verbose=False,**kargs):
    r""" 
    Profile likelihood analysis for uncertainty quantification

    Parameters
    ----------
    model : :ref:`Model`
        Model object describing the data. All non-linear model parameters are profiled by default.

    y : array_like or list of array_like
        Experimental dataset(s).

    args : positional arguments
        Any other positional arguments to be passed to the ``fit`` function. See the 
        documentation of the ``fit`` function for further details. 

    parameters : string or list thereof
        Model parameters to profile. If set to ``'all'`` all non-linear parameters in the model are analyzed. 

    samples : integer scalar
        Number of points to take to estimate the profile function.

    noiselvl : float, optional
        Noise level(s) of the datasets. If set to ``None`` it is determined automatically.

    verbose : boolean, optional
        Specifies whether to print the progress of the bootstrap analysis on the 
        command window, the default is false.

    kargs : keyword-argument pairs
        Any other keyword-argument pairs to be passed to the ``fit`` function. See the 
        documentation of the ``fit`` function for further details. 

    Returns
    -------
    profuq : :ref:`UQResult`
        Profile uncertainty quantification for each non-linear parameter in the model. 
    """

    if noiselvl is None: 
        noiselvl = noiselevel(y)

    # Optimize the whole model to fit the data
    fitresult = fit(model, y, *args, **kargs)

    # Prepare the statistical threshold function
    threshold = lambda coverage: noiselvl**2*chi2.ppf(coverage, df=1) + fitresult.cost

    # Loop over all parameters in the model
    uqresults = {}
    if parameters=='all':
        parameters = model._parameter_list()
    elif not isinstance(parameters,list):
        parameters = [parameters]
    for parameter in parameters:
        if np.any(getattr(model,parameter).linear):
            uqresults[parameter] = None
            continue 

        if verbose:
            tqdm.write(f'Profiling model parameter {parameter}:',end='')
        
        # Construct the values of the model parameter to profile
        start = np.maximum(getattr(model, parameter).lb, getattr(fitresult,parameter)-10*getattr(fitresult,f'{parameter}Uncert').std)
        stop  = np.minimum(getattr(model, parameter).ub, getattr(fitresult,parameter)+10*getattr(fitresult,f'{parameter}Uncert').std)
        varvals = np.linspace(start,stop,samples)

        # Calculate the profile objective function for the parameter
        profile = np.zeros(samples)
        for n,value in enumerate(tqdm(varvals, disable=not verbose)): 

            # Freeze the model parameter at current value
            getattr(model, parameter).freeze(value)

            # Optimize the rest
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitresult_ = fit(model, y, *args, **kargs)

            # Extract the objective function value
            profile[n] = fitresult_.cost

            # Unfreeze the parameter
            getattr(model, parameter).unfreeze()
        profile = {'x':np.squeeze(varvals),'y':profile}
        uqresults[parameter] = UQResult('profile', data=getattr(fitresult,parameter), profiles=profile, threshold=threshold, noiselvl=noiselvl)
        uqresults[parameter].profile = uqresults[parameter].profile[0]
        
    return uqresults