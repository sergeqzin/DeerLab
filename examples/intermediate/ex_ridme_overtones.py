# %% [markdown]
""" 
Analysis of a 5-pulse RIDME signal with dipolar overtones
-------------------------------------------------------------------------

Fit a 5-pulse RIDME trace when the observer spin is coupled to a high-spin center.
It is assumed that the overtone coefficients are fixed, so only global modulation depth is fitted along with the distance distribution.
""" 

import numpy as np 
import deerlab as dl 
import matplotlib.pyplot as plt 

# Data generation (will be ultimately removed)
#=============================================
tau1_gen = 0.4
tau2_gen = 3.2
t_gen = np.linspace(-0.12, tau2_gen, 500) + tau1_gen
r_gen = np.linspace(1, 4, 100)
P_gen = dl.dd_skewgauss(r_gen, 2.5, 0.25, 2)

# plt.plot(r_gen, P_gen)
# plt.show()

overtones_gen = [0.35, 0.2, 0.1]

K_gen = dl.dipolarkernel(t_gen, r_gen, pathways=[{'amp': 1-sum(overtones_gen)},
        {'amp': overtones_gen[0], 'reftime': tau1_gen, 'harmonic': 1},
        {'amp': overtones_gen[1], 'reftime': tau1_gen, 'harmonic': 2},
        {'amp': overtones_gen[2], 'reftime': tau1_gen, 'harmonic': 3}])

B_gen = dl.bg_strexp(t_gen-tau1_gen, 0.3, 1.9) # background

V_gen = (K_gen @ P_gen) * B_gen
V_gen = V_gen + dl.whitegaussnoise(t_gen, 0.015)

# plt.plot(t_gen, V_gen)
# plt.show()

# Data evaluation
#================

path = '../data'
file = 'example_5pridme_overtones.DTA'

tau1 = 0.4
tau2 = 3.2
deadtime = 0.28

t,Vexp = t_gen,V_gen # temporary
# # Load the experimental data
# t,Vexp = dl.deerload(path + file)

# # Pre-processing
# Vexp = dl.correctphase(Vexp) # Phase correction
# Vexp = Vexp/np.max(Vexp)     # Rescaling (aesthetic)
# t = t + deadtime             # Account for deadtime

# Distance vector
r = np.linspace(1, 4, 100) # nm
pulselength = 0.024 # us

# Construct the experiment with overtones
Novertones = 3 # we expect 3 dipolar overtones
Ps = np.array([0.5, 0.4, 0.1]) # normalized overtone coefficients
Ps /= sum(Ps)

# Building an extended RIDME model allowing higher harmonics.
def reftimes(tau1,tau2):
    return [tau1]*Novertones
# Pulse delays 
delays = [tau1,tau2]
# Theoretical dipolar harmonics
harmonics = list(range(1,Novertones+1))
# Harmonics labels
pathwaylabels = np.arange(1,len(harmonics)+1)

my_ridme = dl.ExperimentInfo('5-pulse RIDME with overtones', reftimes, harmonics, pulselength, pathwaylabels, delays)

Vmodel = dl.dipolarmodel(t, r, Bmodel=dl.bg_strexp, experiment=my_ridme)

# Linking refocusing times of different overtones
Vmodel = dl.link(Vmodel, reftime=[f'reftime{n+1}' for n in range(Novertones)])
# Introducing a total modulation depth lam = lam1 + lam2 + lam3
Vmodel.addnonlinear('lam', 0, 1, 0.1, 'Modulation depth', '', 'Overall modulation depth')
# Fixing ratios of overtones
Vmodel = dl.relate(Vmodel, lam1=lambda lam: lam*Ps[0], lam2=lambda lam: lam*Ps[1], lam3=lambda lam: lam*Ps[2])

# Fit the model to the data
results = dl.fit(Vmodel, Vexp)

# Print results summary
print(results)

# Extract fitted dipolar signal
Vfit = results.model

# Extract fitted distance distribution
Pfit = results.P
Pci95 = results.PUncert.ci(95)
Pci50 = results.PUncert.ci(50)

# Extract the unmodulated contribution
Bfcn = lambda lam,decay,stretch,reftime: results.P_scale*(1-lam)*dl.bg_strexp(t-reftime,decay,stretch)
Bfit = results.evaluate(Bfcn)
Bci = results.propagate(Bfcn).ci(95)

plt.figure(figsize=[6,7])
violet = '#4550e6'
plt.subplot(211)
# Plot experimental and fitted data
plt.plot(t,Vexp,'.',color='grey',label='Data')
plt.plot(t,Vfit,linewidth=3,color=violet,label='Fit')
plt.plot(t,Bfit,'--',linewidth=3,color=violet,label='Unmodulated contribution')
plt.fill_between(t,Bci[:,0],Bci[:,1],color=violet,alpha=0.3)
plt.legend(frameon=False,loc='best')
plt.xlabel('Time $t$ (μs)')
plt.ylabel('$V(t)$ (arb.u.)')
# Plot the distance distribution
plt.subplot(212)
plt.plot(r,Pfit,color=violet,linewidth=3,label='Fit')
plt.fill_between(r,Pci95[:,0],Pci95[:,1],alpha=0.3,color=violet,label='95%-Conf. Inter.',linewidth=0)
plt.fill_between(r,Pci50[:,0],Pci50[:,1],alpha=0.5,color=violet,label='50%-Conf. Inter.',linewidth=0)
plt.legend(frameon=False,loc='best')
plt.autoscale(enable=True, axis='both', tight=True)
plt.xlabel('Distance $r$ (nm)')
plt.ylabel('$P(r)$ (nm$^{-1}$)')
plt.tight_layout()
plt.show()