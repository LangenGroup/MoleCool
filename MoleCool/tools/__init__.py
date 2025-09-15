# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 14:18:38 2023

@author: fkogel

v3.5.0

This module contains all different kinds of tools to be used in the other main
modules.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from tqdm import tqdm
import multiprocessing
from copy import deepcopy
import os, json
import pickle #_pickle as pickle
import time
from collections.abc import Iterable
from collections import namedtuple
#%%
def save_object(obj,filename=None):
    """Save an entire class with all its attributes (or any other python object).
    
    Parameters
    ----------
    obj : object
        The object you want to save.
    filename : str, optional
        the filename to save the data. The extension '.pkl' will be added for
        saving the file. If no filename is provided, it is set to the attribute
        `description` of the object and if the object does not have this
        attribute, the filename is set to the name of the class belonging to
        the object.
    """
    if filename == None:
        if hasattr(obj,'description'): filename = obj.description
        else: filename = type(obj).__name__ # instance is set to name of its class
    if type(obj).__name__ == 'System':
        if 'args' in obj.__dict__:
            if 'return_fun' in obj.args:
                del obj.args['return_fun'] #problem when an external function is tried to be saved
        if 'intensity_func_sum' in obj.lasers.__dict__:
            obj.lasers.__dict__['intensity_func_sum'] = None
        if 'intensity_func' in obj.lasers.__dict__:
            obj.lasers.__dict__['intensity_func'] = None
    with open(filename+'.pkl','wb') as output:
        pickle.dump(obj,output,protocol=4)
        
def open_object(filename):
    """Opens a saved object from a saved .pkl-file with all its attributes.    

    Parameters
    ----------
    filename : str
        filename without the '.pkl' extension.

    Returns
    -------
    output : Object
    """
    with open(filename+'.pkl','rb') as input:
        output = pickle.load(input)
    return output

def return_fun_default(system):
    NeNum       = system.levels.uNum
    dic         = dict(
        exectime    = system.exectime,
        photons     = system.photons, #np.squeeze()
        success     = system.success,
        )
    if 'steadystate' in system.args and system.args['steadystate']:
        dic.update(
            F       = system.F.mean(axis=-1),
            Ne      = system.N[-NeNum:,:].sum(axis=0).mean(),
            steps   = system.step+1,
            )
    else:
        tNum    = obj.t.size//10
        dic.update(
            F       = system.F[:,-tNum:].mean(axis=-1),
            Ne      = system.N[-NeNum:,-tNum:].sum(axis=0).mean(),
            )
    
    return dic
    # return dict(system=system)
    
#%%
def get_constants_dict(name=''):
    def openjson(root_dir):
        with open(os.path.join(root_dir, f"{name}.json"), "r") as read_file:
            data = json.load(read_file)
        return data

    if name:
        try:
            return openjson(".")
        except FileNotFoundError:
            try:
                script_dir = os.path.dirname(os.path.abspath(__file__)) #directory where this script is stored.
                constants_dir = os.path.join(os.path.dirname(script_dir), "constants")
                return openjson(constants_dir)
            except FileNotFoundError:
                try:
                    return openjson("./constants/")
                except FileNotFoundError:
                    return openjson("./mymodules/constants/")
    else:
        return {}

def make_axes_invisible(axes,xaxis=False,yaxis=False,
                        invisible_spines=['top','bottom','left','right']):
    """For advanced plotting: This function makes certain properties of an
    matplotlib axes object invisible. By default everything of a new created
    axes object is invisible.

    Parameters
    ----------
    axes : matplotlib.axes.Axes object or iterable of objects
        axes for which properties should be made inivisible.
    xaxis : bool, optional
        If xaxis is made invisible. The default is False.
    yaxis : bool, optional
        If yaxis is made invisible. The default is False.
    invisible_spines : list of strings, optional
        spines to be made invisible. The default is ['top','bottom','left','right'].
    """
    if not isinstance(axes,Iterable): axes = [axes]
    for ax in axes:
        ax.axes.get_xaxis().set_visible(xaxis)
        ax.axes.get_yaxis().set_visible(yaxis)
        for pos in invisible_spines:
            ax.spines[pos].set_visible(False)
            
#%%
def multiproc(obj,kwargs):
    #___problem solving with keyword arguments
    for kwargs2 in kwargs['kwargs']:
        kwargs[kwargs2] = kwargs['kwargs'][kwargs2]
    del kwargs['self']
    del kwargs['kwargs']
    
    #no looping through magnetic field strength or direction for rateeqs so far
    if obj.calcmethod == 'rateeqs': obj.Bfield.reset()
    
    #___recursive function to loop through all iterable laser variables and append result objects
    def recursive(_laser_iters,index):
        if not _laser_iters:
            for i,dic in enumerate(laser_list):
                for key,value in dic.items():
                    # if kwargs['verbose']: print('Laser {}: key {} is set to {}'.format(i,key,value[index[key]]))
                    obj.lasers[i].__dict__[key] = value[index[key]]
                    #or more general here: __setattr__(self, attr_name, value)
            # result_objects.append(pool.apply_async(np.sum,args=(np.arange(3),)))
            if obj.calcmethod == 'OBEs':
                result_objects.append(pool.apply_async(deepcopy(obj).calc_OBEs,kwds=(kwargs)))
            elif obj.calcmethod == 'rateeqs':
                result_objects.append(pool.apply_async(deepcopy(obj).calc_rateeqs,kwds=(kwargs)))
            # print('next evaluation..')
        else:
            for l1 in range(laser_iters_N[ _laser_iters[0] ]):
                index[_laser_iters[0]] = l1
                recursive(_laser_iters[1:],index)
    
    #___Parallelizing using Pool.apply()
    pool = multiprocessing.Pool(obj.multiprocessing['processes'],
                                maxtasksperchild=obj.multiprocessing['maxtasksperchild']) #Init multiprocessing.Pool()
    result_objects  = []

    # identify which parameters to loop through (including Bfield, v0, r0, laser parameters)
    laser_iters_N, laser_list = obj.lasers._identify_iter_params()
    iters_dict      = dict(obj._identify_iter_params(), **laser_iters_N)
    
    #___expand dimensions of v0, r0 in order to be able to loop through them
    r0_arr          = np.atleast_2d(obj.r0)
    v0_arr          = np.atleast_2d(obj.v0)
    #if v0_arr and r0_arr have the same length they should be varied at the same time and not all combinations should be calculated.
    if len(r0_arr) == len(v0_arr) and len(r0_arr) > 1:
        del iters_dict['v0']
    
    #___looping through all iterable parameters of system and laser
    for b1,strength in enumerate(np.atleast_1d(obj.Bfield.strength)):
        for b2,direction in enumerate(np.atleast_2d(obj.Bfield.direction)):
            obj.Bfield.turnon(strength,direction)
            for b3,r0 in enumerate(r0_arr):
                obj.r0 = r0
                for b4,v0 in enumerate(v0_arr):
                    if (len(r0_arr) == len(v0_arr)) and (b3 != b4): continue
                    obj.v0 = v0
                    recursive(list(laser_iters_N.keys()),{})
                    
    if kwargs['verbose']: print('starting calculations for iterations: {}'.format(iters_dict))
    time.sleep(.5)
    # print( [r.get() for r in result_objects])
    # results = [list(r.get().values()) for r in result_objects]
    # keys = result_objects[0].get().keys() #switch this task with the one above?
    results, keys = [], []
    if obj.multiprocessing['show_progressbar']:
        iterator = tqdm(result_objects,smoothing=0.0)
    else:
        iterator = result_objects
    for r in iterator:
        results.append(list(r.get().values()))
    keys = result_objects[0].get().keys() #switch this task with the one above?
    pool.close()    # Prevents any more tasks from being submitted to the pool.
    pool.join()     # Wait for the worker processes to exit.
    
    out = {}
    for i,key in enumerate(keys):
        first_el = np.array(results[0][i])
        if first_el.size == 1:
            out[key] = np.squeeze(np.reshape(np.concatenate(
                np.array(results,dtype=object)[:,i], axis=None), tuple(iters_dict.values())))
        else:
            out[key] = np.squeeze(np.reshape(np.concatenate(
                np.array(results,dtype=object)[:,i], axis=None), tuple([*iters_dict.values(),*(first_el.shape)])))
    
    Results = namedtuple("Results_OBEs_rateeqs", ["vals", "iters"])
    return Results(out, iters_dict)

#%%
def vtoT(v,mass=157):
    """function to convert a velocity v in m/s to a temperatur in K."""
    from scipy.constants import k, u
    return v**2 * (mass*u)/k #v**2 * 0.5*(mass*u)/k

def Ttov(T,mass=157):
    """function to convert a temperatur in K to a velocity v in m/s."""
    from scipy.constants import k, u
    return np.sqrt(k*T/(mass*u)) #np.sqrt(k*T*2/(mass*u))

def gaussian(x, a=1.0, x0=0.0, std=1.0, y_off=0):
    """Standard Gaussian function :math:`a \exp(-0.5(x-x_0)^2/std^2)+y_{off}`."""
    return a * np.exp(-0.5 * ((x - x0) / std)**2) + y_off

def FWHM2sigma(FWHM):
    """Convert full width at half maximum (FWHM) to standard deviation
    of a Gaussian (sigma)."""
    return FWHM/2.3548200450309493

def sigma2FWHM(sigma):
    """Convert standard deviation of a Gaussian (sigma) to full width
    at half maximum (FWHM)."""
    return sigma*2.3548200450309493

#%%
def get_results(fname, Z_keys='F', XY_keys=[], XY_data_fmt={}, XYY_inds=[],
                add_v0=False, add_flip_v=False, add_I0=False, scale_F='N'):
    """Extract results from an .pkl file as observables of a high-dimensional
    parameter space in a certain order of parameters and their values as numpy
    arrays. Optionally, one can add e.g. zero force values, manually.

    Parameters
    ----------
    fname : str or ~System.System
        filename where the instance of System is saved or the instance itself.
    Z_keys : str, optional
        Observable calculated in the results. The default is 'F'.
    XY_keys : list of str, optional
        Parameters that the results are dependent on, e.g. ['v0','I']. The default is [].
    XY_data_fmt : dict of func, optional
        Dictionary with parameters as keys and functions as dict values which
        determine how the parameter values are converted or formatted, e.g. 
        {'strength': lambda system: system.Bfield.strength}. The default is {}.
    XYY_inds : list of inds, optional
        list of indices to choose values of further parameters included in the
        results dataset for more than 3 dimensions. The default is [].
    add_v0 : bool, optional
        Manually adding velocity v=0 and zero force. The default is True.
    add_flip_v : bool, optional
        add a flip velocity axis to get the full range. The default is True.
    add_I0 : bool, optional
        adding intensity I=0 where the force is zero. The default is True.
    scale_F : str, optional
        scale F to either 'N' or 'hbar*k*Gamma/2'. Do nothing if scale_F=='.',
        The default is 'N'.

    Returns
    -------
    Z : np.ndarray
        observables or results as shape of both axes of XY_keys.
    XY : dict
        names and data of X axis and Y axis as str and np.ndarray
        (length of both axis together equal the shape of Z).
    XYY : dict
        key and single values of the parameters of further parameters included
        in the results dataset for more than 3 dimensions.
    """
    # Format values for iterating parameters
    XY_data_fmt_default = {'I': lambda x: x.lasers.I_sum,
                           'v0': lambda x: x.v0[:,2],
                           'strength': lambda x: x.Bfield.strength}
    XY_data_fmt_default.update(XY_data_fmt)
    XY_data_fmt     = XY_data_fmt_default
    
    # loading system instance
    if isinstance(fname, str):
        s4  = open_object(fname)
        fname_bn= os.path.basename(fname)
        print(f"File <{fname_bn}> with iteration variables: {s4.results[1]}")
    else:
        s4  = fname
        print(f"Instance <{fname_bn}> loaded with iteration variables: {s4.results[1]}")
    
    # iterating over multiple Z_keys:
    if isinstance(Z_keys, str): Z_keys = [Z_keys]
    Z_dict = dict()
    
    for Z_key in Z_keys:
        
        # initiating the parameters that are shown on the X and Y axes
        iterinds    = dict(zip(s4.results[1].keys(),range(len(s4.results[1].keys()))))
        iterinds_keys = list(iterinds.keys())
        if not XY_keys:
            if 'v0' in iterinds_keys:
                iterinds_keys.remove('v0')
                XY_keys = ['v0',iterinds_keys[0]]
            else:
                XY_keys = iterinds_keys[:2]
        else:
            for XY_key in XY_keys:
                if XY_key not in iterinds:
                    raise ValueError(f'Value <{XY_key}> not included in results from file {fname_bn}!')
            if len(XY_keys) != 2:
                iterinds_keys.remove(XY_keys[0])
                XY_keys.append(iterinds_keys[0])
        
        # actual X,Y axes data arrays
        XY_data = []
        for XY_key in XY_keys:
            if XY_key not in XY_data_fmt:
                raise Exception(f'XY_data_fmt must be given for XY_key {XY_key}')
            XY_data.append(np.array(XY_data_fmt[XY_key](s4)))
    
        # loading the Z data (can be 2D or high-dimensional)
        if Z_key not in s4.results[0].keys():
            raise ValueError("Z_key '{}' not included in results, try one of {}".format(
                Z_key, list(s4.results[0].keys())))
            
        Z = s4.results[0][Z_key].transpose((iterinds.pop(XY_keys[0]),
                                            iterinds.pop(XY_keys[1]),
                                            *iterinds.values()))
        
        # values of further parameters in dataset if Z has more than 2 dimensions
        XYY = dict()
        if len(iterinds) != 0: # pick single value of the parameter of higher dimensions
            if not XYY_inds:
                XYY_inds = [0]*len(iterinds)     
            elif len(XYY_inds) != len(iterinds):
                raise ValueError(f'len of XYY_inds not equal to number of further dims ({len(iterinds)})')    
        
            for XYY_key, XYY_ind in zip(iterinds.keys(), XYY_inds):
                XYY[XYY_key] = XY_data_fmt[XYY_key](s4)[XYY_ind]
            
            for XYY_ind in XYY_inds[::-1]:
                Z = Z[...,XYY_ind]
    
        # add manually zero forces e.g. for zero intensity or zero velocity    
        if (add_flip_v or add_v0) and ('v0' in XY_keys) and Z_key in ['F','Ne']:
            axis    = XY_keys.index('v0')
            Z       = np.moveaxis(Z, axis, 0) # move old axis to zero axis
            v_arr   = XY_data[axis]
            if add_flip_v:
                v_arr   = np.array([*v_arr,*(-np.flip(v_arr))])
                if Z_key == 'F':
                    Z       = np.array([*Z,*(-np.flip(Z,axis=0))])
                elif Z_key == 'Ne':
                    Z       = np.array([*Z,*(+np.flip(Z,axis=0))])
            if add_v0:
                v_arr   = np.array([*v_arr,0])
                Z       = np.array([*Z, 0*Z[0]])
            v_inds  = np.argsort(v_arr)
            Z       = Z[v_inds]
            Z       = np.moveaxis(Z, 0, axis) # move zero axis back to old axis
            XY_data[axis] = v_arr[v_inds]
        
        if add_I0 and ('I' in XY_keys):
            axis    = XY_keys.index('I')
            Z       = np.moveaxis(Z, axis, 0) # move old axis to zero axis
            I_arr   = XY_data[axis]
            Z       = np.array([0*Z[0], *Z])
            Z       = np.moveaxis(Z, 0, axis) # move zero axis back to old axis
            XY_data[axis] = np.array([0.,*I_arr])
            
        if scale_F and (Z_key == 'F'):
            # Further scaling of the Z data, i.e. the force to more intuitive unit
            from System import hbar, pi
            Gamma   = s4.levels.calc_Gamma().mean()
            lamb    = s4.lasers.getarr('lamb').mean()
            unit    = hbar*2*pi/lamb*Gamma/2
            if (scale_F == 'hbar*k*Gamma/2') and (abs(Z.max()) < 100*unit):
                Z /= unit
            elif (scale_F == 'N') and (abs(Z.max()) > 100*unit):
                Z *= unit
            else:
                print('No Scaling of the force applied.')
    
        Z_dict[Z_key] = Z
    
    if not np.all(np.array([arr.shape for arr in Z_dict.values()])==Z_dict[Z_keys[0]].shape):
        raise Exception('Shapes of generated Z data is not the same for every Z_key!')
        
    return Z_dict, dict(zip(XY_keys,XY_data)), XYY
    
def plot_results(fname, Z_keys=['F'], XY_data_fmt={}, scale_F='hbar*k*Gamma/2',
                 XY_labels={}, Z_labels={}, cmap='RdBu', levels = 12,
                 Xlim=[], Ylim=[], Zlim={}, Z_percent=['F','Ne'],
                 figname='', savefig=True, **kwargs):
    """plot results for calculating one or multiple observables in a high-dimensional
    parameter space.

    Parameters
    ----------
    fname : str or ~System.System
        see function get_results.
    Z_keys : str or list of str, optional
        names of calculated observables to be plotted (see function get_results).
        The default is ['F'].
    XY_data_fmt : dict, optional
        see function get_results. The default is {}.
    XY_labels : dict, optional
        dictionary to convert the names of certain parameters to other more suitable
        axis labels. The default is {}.
    Z_labels : dict, optional
        dictionary to convert the names of the observables to other more suitable
        axis labels. The default is {}.
    cmap : str, optional
        color map from matplotlib. The default is 'RdBu'.
    levels : int, optional
        number of color levels. The default is 12.
    Xlim : tuple, optional
        lower and upper limit. The default is [].
    Ylim : tuple, optional
        lower and upper limit. The default is [].
    Zlim : tuple, optional
        lower and upper limit. The default is {}.
    Z_percent : list, optional
        list with the names of observables to be plotted in percent.
        The default is ['F','Ne'].
    figname : str, optional
        name of the figure. The default is ''.
    savefig : bool or str, optional
        if True the figure is saved. If str, the figure is saved using the
        provided string. The default is True.
    **kwargs : keyword arguments
        keyword arguments for the function get_results.
    """
    # defining some default dictionaries for axes labels, and formatters for the axes data
    Z_labels_default = dict(F='Cooling force $F$ ($\hbar k \Gamma/2$)',
                            Ne='Ex. state population $n_e$',
                            exectime='Execution time (s)',
                            steps='Iter. steps till steady state')
    
    XY_data_fmt_default = {'I': lambda x: x.lasers.I_sum/1000,
                           'strength': lambda x: x.Bfield.strength*1e4}
    XY_labels_default = {'I':'Intensity $I_{tot}$ (kW/m$^2$)',
                         'v0':'Velocity $v$ (m/s)',
                         'strength':'Bfield strength (G)'}
    Z_labels_default.update(Z_labels)
    XY_labels_default.update(XY_labels)
    XY_data_fmt_default.update(XY_data_fmt)
    Z_labels        = Z_labels_default
    XY_labels       = XY_labels_default
    XY_data_fmt     = XY_data_fmt_default
    
    if isinstance(Z_keys, str): Z_keys = [Z_keys] # make sure that Z_keys is a list of str
    
    # Initializing figure
    plt.rcParams['figure.constrained_layout.use'] = False
    fig, axs = plt.subplots(len(Z_keys),1, sharex=True, figsize=(8,7), squeeze=False,
                            num=figname if figname else None)
    fig.subplots_adjust(hspace=0.0)
    
    # iterating over keys for Z (actual results), e.g. Force F, excited state fraction Ne
    for i,(ax,Z_key) in enumerate(zip(axs[:,0],Z_keys)):
        
        # load results and make meshgrid for plotting
        Z_dict, XY, XYY = get_results(fname, Z_keys=Z_key, scale_F=scale_F,
                                      XY_data_fmt=XY_data_fmt,**kwargs)
        Z       = Z_dict[Z_key]
        X,Y     = np.meshgrid(*XY.values())
        
        # update axes labels for undefined values:
        for XY_key in [*list(XY.keys()),*(XYY.keys())]:
            if XY_key not in XY_labels:
                XY_labels.update({XY_key:XY_key})
        
        if Z_key in Z_percent: 
            Z *= 1e2 # in percent
            
        # ======== PLOTTING ========
        # set title
        if i == 0 and len(XYY) != 0:
            title = ',\n'.join([f'{XY_labels[key]}: {data:4g}'
                                for key,data in XYY.items()])
            ax.set_title(title)
        # set x, y labels
        if i == len(Z_keys)-1:
            ax.set_xlabel(XY_labels[list(XY.keys())[0]])
        if i == len(Z_keys) //2:
            ax.set_ylabel(XY_labels[list(XY.keys())[1]])
        # set axes limits
        if Xlim:    ax.set_xlim(*Xlim)
        if Ylim:    ax.set_ylim(*Ylim)
        vmin,vmax = None, None
        if Z_key in Zlim:
            vmin, vmax= Zlim[Z_key]
        # draw 2D countour data and colourbar
        CS = ax.contourf(X,Y,Z.T,levels=levels,cmap=cmap,vmin=vmin,vmax=vmax)#np.linspace(1.,2.,11))
        CS2= ax.contour(CS, levels=[0.0], colors='k', origin='lower',linestyles='dashed')
        cbar = plt.colorbar(CS, ax=ax, aspect=14, pad=0.01, shrink=0.90)
        cbar.add_lines(CS2)
        cbar.ax.set_ylabel(Z_labels[Z_key])
        if Z_key in Z_percent: # showing Z data in percent
            cbar.ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    
    # saving figure    
    if savefig:
        if isinstance(savefig, str):
            figname = savefig
        elif not figname:
            figname = f"{fname}_{'-'.join(Z_keys)}"
        plt.savefig(figname)