# -*- coding: utf-8 -*-
"""
This Module contains different type of functions and classes, e.g. to calculate
simple linear trajectories through multiple apertures and to evaluate trajectories
as solutions of the Monte Carlo simulations.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import h5py
from . import Ttov, vtoT, gaussian, FWHM2sigma, sigma2FWHM

#%%
def transversal_width2Temp(w_sigma=1.2e-3, v_Fw=170, total_dist=58e-2, mass=157,
                           printing=True):
    """Calculate the velocity and corresponding temperature of a particle that
    is flying for a certain total distance and finally reaches a specific
    transversal position starting from zero position. The temperature corresponds
    to a transversal Gaussian distribution with standard deviation given by the
    transversal position `w_sigma`.

    Parameters
    ----------
    w_sigma : float or np.ndarray, optional
        transversal position in m. The default is 1.2e-3.
    v_Fw : float, optional
        Forward velocity in m/s. The default is 170.
    total_dist : float, optional
        Total flying distance in m. The default is 58e-2.
    mass : float, optional
        atomic mass units of the particle. The default is 157.
    printing : bool, optional
        for enabling printing additional information. The default is True.

    Returns
    -------
    T : float or np.ndarray
        temperature in K.
    v_tr : float or np.ndarray
        Transversal velocity  in m/s.
    """
    
    # time of flight from buffer gas output to camera position
    t       = total_dist/v_Fw
    # transversal velocity correpsonding to certain transversal position (standard deviation sigma)
    v_tr    = w_sigma/t
    # converting into temperature
    T       = vtoT(v_tr,mass=157)
    if printing:
        print(f"T={T*1e3:.4g} mK, v={v_tr:.3f} m/s")
    return T, v_tr

#%%
def load_init_distr(system, fname, samplesize=None, x_init=0., plotting=False):
    """loading initial velocity and position distribution into system instance.

    Parameters
    ----------
    system : ~MoleCool.System.System
        The System into which the distributions should be stored.
    fname : str
        file name of the distribution pkl file.
    samplesize : int, optional
        number of samples to be used from the distribution.
        The default is None for taking all samples.
    x_init : int, optional
        Distance from x=0 where initial free propagation starts to let the initial
        distribution evolve in the transverse axes.
        Or, initial free propagation until entering interaction zone.
        At x=0 The MC trajectory simulation starts. The default is 0..
    plotting : bool, optional
        Whether plotting is enabled. The default is False.
    """
    sampleslice = slice(samplesize) # None for all or int number for smaller number of samples
    
    # initial distributions normally at the output aperture of the buffergas cell
    with h5py.File(fname,'r') as h5file:
        rdist = h5file['dists/rdist'][:][sampleslice]
        vdist = h5file['dists/vdist'][:][sampleslice]

    # initial free propagation until entering interaction zone. x=0 remains the same
    rdist[:,1:] += vdist[:,1:]*x_init/vdist[:,0][:,None]
    
    system.v0, system.r0 = vdist, rdist
    system.x_init = x_init
    
    if plotting:
        plt.figure('vel dist y')
        plt.hist(vdist[:,1],bins=30)
        plt.figure('pos dist y')
        plt.hist(rdist[:,1]*1e3, bins=30)

#%%
class TrajectoryApertures():
    def __init__(self, name='TrajApers', diameter0=3e-3,
                 diameters=[5e-3, 3.3e-3, 10, 10],
                 x_aper=[28e-3, 123e-3, 170e-3, 550e-3],
                 labels=['aper1', 'aper2', 'PD1', 'PD2']):
        """Set up an instance for calculating linear propagation trajectories
        through multiple round apertures. To initiate, specify the apertures'
        labels and geometrics.
        
        The whole simulation is based on the following default axes definitions:
            
            - x-axis: longitudinal direction (molecular beam axis)
            - y-axis: transversal horizontal direction along probe lasers
            - z-axis: transversal vertical direction

        Parameters
        ----------
        name : str, optional
            with this you can name the instance which is used in other functions
            for saving figures or hdf5 files if no other filename is specified.
            Default is 'TrajApers'.
        diameter0 : float, optional
            Diameter of buffer gas cell output in m. The default is 3e-3.
        diameters : list, optional
            Diameters of the apertures in m. The default is [5e-3, 3.3e-3, 10, 10].
        x_aper : list, optional
            Positions of the apertures along the x-axis.
            The default is [28e-3, 123e-3, 170e-3, 580e-3].
        labels : list, optional
            Labels of the apertures. The default is ['aper1', 'aper2', 'PD1', 'PD2'].
        """
        self.name       = name
        self.radii      = np.array([diameter0, *diameters])/2
        self.x_aper     = np.array([0., *x_aper])
        self.labels     = ['Start', *labels]
        
        # empty lists for storing the distributions for all apertures
        self.rdists     = []
        self.vdists     = []
        
        self.hist_data  = dict(r={}, v={})
        
    def calc_propagation_apers(self, samplesize=5000000,
                               mu=[200, 0, 0], sigma=[32., 12., 12.],
                               v_tr_max=0.):
        """Calculate the distributions at every aperture using simple linear
        trajectories of the particles that are removed when hitting on the apertures.

        Parameters
        ----------
        samplesize : int, optional
            how many particles or samples trajectories to be calculated.
            The default is 5000000.
        mu : list, optional
            mean values of all 3 axes in m/s for initializing a 3D Gaussian velocity
            distribution at the start. The default is [200, 0, 0].
        sigma : list, optional
            Same as `mu` but standard deviations. The default is [32., 12., 12.].
        v_tr_max : float, optional
            maximum initial transversal velocity at the edge of the buffer gas
            cell output. The transversal velocity distributions' mean is shifted
            linearly from 0 to v_tr_max at the edges. The default is +0.
        """
        samplesize      = int(samplesize)
        self.mu         = np.array(mu)
        self.sigma      = np.array(sigma)
        self.v_tr_max   = v_tr_max
        
        #%% initial distributions
        # uniform transversal position distribution 2D:
        # base = 6
        # rands   = (base**np.random.rand(samplesize)-base**0)/(base**1-base**0)
        rands   = np.random.rand(samplesize)
        length  = np.sqrt(rands) *self.radii[0] #np.sqrt(np.rand...) for uniformly distributed points
        angle   = 2*np.pi * np.random.rand(samplesize)
        y0,z0   = length * np.cos(angle), length * np.sin(angle)
        rdist0  = np.array([ np.zeros(samplesize), y0, z0 ]).T
        
        # Gaussian velocity distribution 3D
        theta   = np.arctan2(rdist0[:,2],rdist0[:,1]) # arcus tangens(y,x) that is not periodic in pi but 2pi
        mu_arr  = np.array([np.zeros(samplesize), np.cos(theta), np.sin(theta)]).T # mean velocity for Gaussian vel. distr.
        mu_arr[:,1:] *= (np.linalg.norm(rdist0[:,1:],axis=-1)/self.radii[0]*self.v_tr_max)[:,None] # linear increase of mean velocity from center to edge of buffer gas cell output
        mu_arr += self.mu[None,:] # shift the velocity distribution by a constant offset for each direction
        vdist0  = np.random.normal(loc=mu_arr, scale=self.sigma) # crate vel. distr.
        
        # Gaussian velocity distribution 1D which has position dependent mean values
        self.rdists.append(rdist0)
        self.vdists.append(vdist0)
        
        #%% free propagation steps through apertures
        for i,(radius,x_i) in enumerate(zip(self.radii[1:], self.x_aper[1:])):
            rdist       = self.rdists[-1].copy()
            vdist       = self.vdists[-1].copy()
            
            dx          = x_i - rdist[0,0]
            rdist[:,1:] += vdist[:,1:] * dx / vdist[:,0][:,None]
            rdist[:,0]  += dx
            
            lost_mols   = np.where((np.linalg.norm(rdist[:,1:],axis=-1) > radius))[0]
            rdist       = np.delete(rdist,lost_mols,axis=0)
            vdist       = np.delete(vdist,lost_mols,axis=0)
        
            # append distributions from last aperture at the end of array
            self.rdists.append(rdist)
            self.vdists.append(vdist)
        
    def initial_rdist_from_other(self, ind=-1, save_hdf5=False, fname=''):
        """Saving initial distribution of only the molecules that made it
        through all the apertures until reaching the aperture with index ind.

        Parameters
        ----------
        ind : int, optional
            Index of the aperture from which the initial distribution is to be
            calculated. The default is -1.
        save_hdf5 : bool, optional
            Whether to save the initial distribution as hdf5. The default is False.
        fname : str, optional
            filename. The default is ''.

        Returns
        -------
        rdist0_new : np.ndarray
            Calculated new initial distribution from final distribution.
        """
        rdist0_new       = self.rdists[ind].copy()
        rdist0_new[:,1:] -= self.vdists[ind][:,1:] * self.rdists[ind][0,0] / self.vdists[ind][:,0][:,None]
        rdist0_new[:,0]  -= self.rdists[ind][0,0]
        
        if save_hdf5:
            with h5py.File(self.get_fname(fname,end='.hdf5'),'w') as h5file:
                for key in ['radii','x_aper','mu','sigma']:
                    h5file.create_dataset(f"{key}", data=self.__dict__[key])
                h5file.attrs['v_tr_max'] = self.v_tr_max
                h5file.create_dataset("dists/rdist", data=rdist0_new)
                h5file.create_dataset("dists/vdist", data=self.vdists[ind])        
        
        return rdist0_new

    def get_hist_data(self, i, w, r_v='r', bins=30):      
        """Calculate histogram data for the distribution at a certain aperture.
        All important evaluated arrays are saved in the dictionary hist_data.
        
        Parameters
        ----------
        i : int
            index of the aperture.
        w : float
            Should imitate the imaging laser beam width as a width of the slice
            which is cut out of the 2D plane. So, it is the standard
            deviation in m of the Gaussian integrating over z-axis.
        r_v : str, optional
            position of velocity histogram ('r' or 'v'). The default is 'r'.
        bins : int, optional
            Number of bins. The default is 30.
            
        Returns
        -------
        dict
            Dictionary with x and y axes of histogram data (`x_data`, `y_data`),
            fit data (`x_data_fit` and `y_data_fit`) and fit values (`popt`).
            This dictionary is also stored and can afterwards be accessed in the
            attribute ``hist_data`` in the following way: hist_data[r_v][i].
        """
        rdist = self.rdists[i] # position distribution at aperture i
        if r_v == 'r':      dist = rdist
        elif r_v == 'v':    dist = self.vdists[i]
        
        # estimating how many particles are within the standard deviation
        inds    = np.where(np.abs(rdist[:,2]) < w)[0]
        print(i,inds.shape[0]/rdist.shape[0])
        weights = gaussian(rdist[:,2], std=w) # Gaussian weighting
        
        y_data, x_bins = np.histogram(dist[:,1], bins=bins, weights=weights)
        x_data  = (x_bins[1:] + x_bins[:-1]) / 2
        
        # fitting:
        # Initial guess and bounds for the parameters
        p0          = [np.max(y_data), np.mean(x_data), np.std(x_data)]
        bounds      = ([0, np.min(x_data), 0],
                       [np.max(y_data)*10, np.max(x_data) ,np.std(x_data)*100])
        popt, pcov  = curve_fit(gaussian, x_data, y_data, p0=p0, bounds=bounds)
        
        # fitted data arrays
        x_data_fit  = np.linspace(np.min(x_data), np.max(x_data), 500)
        y_data_fit  = gaussian(x_data_fit, *popt)
        
        self.hist_data[r_v][i] = dict(x_data=x_data, y_data=y_data, popt=popt, x_bins=x_bins,
                                      x_data_fit=x_data_fit, y_data_fit=y_data_fit)
        return self.hist_data[r_v][i]

    def plot_pos_vel_distr(self, w=1e-3, show_which=None, bins=(30,30),
                           save_fig=False, fname=''):
        """Plot 1D and 2D position and velocity distributions as histograms.

        Parameters
        ----------
        w : float, optional
            width see :meth:`get_hist_data`. The default is 1e-3.
        show_which : list of int, optional
            indices (of apertures below) for which the velocity distrs should
            be plotted. The default is None.
        bins : tuple, optional
            Numbers of bins for position and velocity histograms, respectively.
        save_hdf5 : bool, optional
            Whether to save the initial distribution as hdf5. The default is False.
        fname : str, optional
            filename. The default is ''.
        """
        
        if not show_which:
            show_which = np.arange(0, len(self.rdists), 1)
        
        # plotting position distributions (histograms) 1D and 2D
        fig, axs = plt.subplots(3,len(show_which),sharey=False,figsize=(12,7))
        
        # 2D histograms --------------------------------------------------------------
        for i,ax in zip(show_which,axs[1]):
            rdist = self.rdists[i]
            ax.hist2d(rdist[:,1]*1e3,rdist[:,2]*1e3,bins=(bins[0],bins[0]),cmap='Blues')
            ax.set_aspect('equal', 'box')
            ax.axhline(0,color='yellow')
            ax.axvline(0,color='yellow')
        
        # 1D histograms --------------------------------------------------------------
        # position distr:
        for i,ax in zip(show_which,axs[0]):
            hist_data   = self.get_hist_data(i, w=w, r_v='r', bins=bins[0])
            density     = self.rdists[i].shape[0]/len(self.rdists[0])
            ax.stairs(hist_data['y_data'], hist_data['x_bins']*1e3,
                      ls='-')
            ax.plot(hist_data['x_data_fit']*1e3, hist_data['y_data_fit'],
                    '--', label='Gaussian Fit', color = 'k', lw = 1)
            ax.set_xlabel('Position $y$ in mm')
            ax.set_title(f"{self.labels[i]}:\n{self.x_aper[i]*1e3} mm\n{density*1e2:.4f}%")
            
        # velocity distr:
        for i,ax in zip(show_which,axs[2]):
            hist_data   = self.get_hist_data(i, w=w, r_v='v', bins=bins[1])
            ax.stairs(hist_data['y_data'], hist_data['x_bins'],
                      ls='-', fill=True)
            ax.plot(hist_data['x_data_fit'], hist_data['y_data_fit'],
                    '--', label='Gaussian Fit', color = 'k', lw = 1)
            ax.set_xlabel('Velocity $v$ in m/s')
        
        plt.subplots_adjust(wspace=0.08)
        fig.suptitle(f"Init. condition: v_tr_max={self.v_tr_max}, mu={self.mu} m/s, sigma={self.sigma} m/s, w={w*1e3:.2f}mm",
                     y=1.05)
        
        # saving figure
        if save_fig: plt.savefig(self.get_fname(fname,'_dists','.png'))
    
    def get_fname(self,fname,middle='',end=''):
        """Add some strings for making filenames"""
        if not fname:   return self.name + middle + end
        else:           return fname + end

    def plot_trajectories(self, N=200, ind=-1, yunit=1e-3, ylim=[-15,15],
                          save_fig=False, fname=''):
        """Plotting trajectories of a certain distribution at a certain aperture.

        Parameters
        ----------
        N : int, optional
            number of trajectories to be plotted. The default is 200.
        ind : int, optional
            Index of the aperture from which the initial distribution is to be
            calculated. The default is -1.
        yunit : float, optional
            unit of y-axis. The default is 1e-3.
        ylim : list, optional
            limits of y axis. The default is [-15,15].
        save_hdf5 : bool, optional
            Whether to save the initial distribution as hdf5. The default is False.
        fname : str, optional
            filename. The default is ''.
        """
        rdist0_new = self.initial_rdist_from_other(ind=ind, save_hdf5=False)
        rdist = self.rdists[ind]
        plt.figure()
        for i in np.arange(N):
            plt.plot([rdist0_new[i,0], rdist[i,0]], [rdist0_new[i,1]/yunit, rdist[i,1]/yunit],
                     '-',c='C0',alpha=0.4)
        # plot apertures:
        ls_apers = {'color':'k','lw':2}
        for radius,x_aper,label in zip(self.radii,self.x_aper,self.labels):
            plt.vlines(x_aper, ymin=radius/yunit, ymax=radius*2/yunit, **ls_apers)
            plt.vlines(x_aper, ymax=-radius/yunit, ymin=-radius*2/yunit, **ls_apers)
        plt.xlabel('Position $x$ in m')
        plt.ylabel('Position $y$ in mm')
        if ylim:
            plt.ylim(ylim)
        
        if save_fig: plt.savefig(self.get_fname(fname,'_traj','.png'))

#%% handling and plotting of calculated Monte Carlo simulation results
def get_dist(system, key='r', unperturbed=False, radial=False,
             x_fly=0, xyz='z'):
    """Extract the results of the Monte Carlo (MC) simulation and calculate the
    distribution after a certain time of flight distance.

    Parameters
    ----------
    system : ~MoleCool.System.System
        System instance from which the distributions are to be loaded.
    key : str, optional
        Type of distribution. Can be 'r' for position, 'v' for velocity and
        'photons' for scattered photon number. The default is 'r'.
    unperturbed : bool, optional
        Whether to neglect the MC results and only use the unperturbed initial
        distribution. The default is False.
    radial : bool, optional
        Whether to calculate a radial distribution using the y and z axes.
        The default is False.
    x_fly : float, optional
        Time of flight distance to propagate the distribution from MC further.
        The default is 0.
    xyz : str, optional
        axis of the disibution to be returned. The default is 'z'.

    Returns
    -------
    np.ndarray
        Position or velocity distribution along a certain axis or scattered
        photon number distribution.
    """
    # if (not 'sols_end' in system.__dict__) and ('sols' in system.__dict__):
    #     system.sols_end = dict()
    #     for key_i,indices in dict(photons=-1, r=[3,4,5], v=[0,1,2]).items():
    #         system.sols_end[key_i] = np.array([sol.y[indices,-1] for sol in system.sols])
    
    final_values = system.trajectory_results['final_values']
    
    if key == 'photons':
        return final_values['photons']
    
    x_init  = system.x_init
    ixyz     = {'x':0,'y':1,'z':2}[xyz]
    
    if unperturbed:
        v       = system.v0
        r       = system.r0.copy()
        x_fly   = x_fly + system.lasers.retrorefl_beams_kwargs['int_length']
    else:
        v       = final_values['v'].copy()
        r       = final_values['r'].copy()
    
    r[:,0]  += x_fly
    r[:,1:] += v[:,1:] * x_fly / v[:,0][:,None]
    
    arr = dict(r=r, v=v)[key]
    if radial:
        return np.sqrt(arr[:,1]**2 + arr[:,2]**2)
    else:
        return arr[:,ixyz]
        
def get_hist_vals(system, yz='z', w=0, radial=False,
                  bins_per_unit=None, xrange=None, **kwargs):
    """Get histogram values from Monte Carlo simulation distributions.

    Parameters
    ----------
    system : ~MoleCool.System.System
        System instance from which the distributions are to be loaded..
    yz : str, optional
        Axis for which the histogram should be calculated. The default is 'z'.
    w : float or dict, optional
        Parameter(s) for weighting along the other transversal axis.    
        Should imitate the imaging laser beam width as a width of the slice
        which is cut out of the 2D plane. So, when a float number is provided,
        it is the standard deviation in m of the Gaussian integrating over z-axis.
        If otherwise a dictionary is provided, it can contain all parameters
        corresponding to the gaussian function (:func:`tools.gaussian`).
        The default is 0.
    radial : bool, optional
        Whether to use a radial distribution. The default is False.
    bins_per_unit : float, optional
        Number of bins per unit (e.g. m or m/s). The default is None.
    xrange : tuple, optional
        xrange for evaluation the bins of the histogram. The default is None.
    **kwargs : kwargs
        further keyword arguments for :meth:`get_dist` function.

    Returns
    -------
    hist_y : np.ndarray
        heights of the histogram's bins.
    hist_x : np.ndarray
        center positions of the histogram's bins.
    """
    # if xrange is provided, calculate number of bins of histogram
    if xrange and bins_per_unit:
        Nbins   = int(bins_per_unit*(max(xrange)-min(xrange))) + 1
    else:
        Nbins   = 30 #default
    
    values  = get_dist(system, radial=radial, xyz=yz, **kwargs)
    
    if radial:
        weights = 1/values
    elif w:
        yz_     = dict(y='z',z='y')[yz] # other transversal axis, i.e. switch y and z axes
        kwargs_weights = {k:v for k,v in kwargs.items() if k not in ['key']}
        values_weights  = get_dist(system, key='r', xyz=yz_, radial=radial,
                                         **kwargs_weights)
        if not isinstance(w, dict):
            w = dict(std=w)
        weights = gaussian(values_weights, **w)
    else:
        weights = None
        
    hist_y,bins = np.histogram(values, weights=weights, bins=Nbins, range=xrange)
    
    hist_x      = (bins[1:]+bins[:-1])/2 # for lineplot istead of step-function
    
    return hist_x, hist_y
    
def plot_hist(system, key='r', yz='z', y_off=0,
              ax=None, color=None, scale_x=1, **kwargs):
    """Plotting histograms from get_hist_vals.
    
    Parameters
    ----------
    system : ~MoleCool.System.System
        See :meth:`TrajectoryApertures.get_hist_data`.
    key : str, optional
        See :meth:`TrajectoryApertures.get_hist_data`. The default is 'r'.
    yz : str, optional
        See :meth:`TrajectoryApertures.get_hist_data`. The default is 'z'.
    y_off : float, optional
        Offset on y-plotting-axis. The default is 0.
    ax : plt.axis, optional
        axis instance where the data is drawn on. The default is None.
    color : str, optional
        Color of the histogram. The default is None.
    scale_x : float, optional
        Value to scale the x-axis for plotting. The default is 1.
    **kwargs : kwargs
        keyword arguements for :meth:`get_hist_vals`.
    """
    if not ax: ax = plt.gca()

    # iterate through unperturbed and perturbed data arrays (forces on/off)
    for color_,label,unperturbed in zip([color,'grey'],
                                        ['lasers on','lasers off'],
                                        [False, True]):
        
        if key == 'photons' and unperturbed: continue

        hist_x, hist_y = get_hist_vals(system, key=key, yz=yz,
                                       unperturbed=unperturbed, **kwargs)
        # plotting
        ax.plot(hist_x*scale_x, hist_y+y_off, color=color_, label=label)
    
    if key == 'photons':
        xlabel = 'Scattered photons'
    else:
        xlabel = dict(r='Position',v='Velocity')[key] \
                   + f" ${key}_{yz}$ (" + dict(r='mm',v='m/s')[key]+')'
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Number of molecules')
    ax.legend()

#%%
if __name__ == '__main__':
    traj = TrajectoryApertures(diameter0=3e-3, name='testApers',
                                diameters=[5e-3, 3.3e-3, 10, 10],
                                x_aper=[28e-3, 123e-3, 170e-3, 550e-3],
                                labels=['aper1', 'aper2', 'PD1', 'PD2'])
    traj.calc_propagation_apers(mu=[170, 0, 0], sigma=[32., *(np.array([1.,1.])*12)])
    traj.plot_pos_vel_distr()
    traj.plot_trajectories()
    traj.initial_rdist_from_other()
    
    print(traj.hist_data['r'][4]['popt'][2]*1e3)