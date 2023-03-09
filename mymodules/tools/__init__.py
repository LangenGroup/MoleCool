# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 14:18:38 2023

@author: fkogel

v3.1.0

This module contains all different kinds of tools to be used in the other main
modules.
"""
import numpy as np
from tqdm import tqdm
import multiprocessing
from copy import deepcopy
import os, json
import _pickle as pickle
import time
from collections.abc import Iterable
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
    with open(filename+'.pkl','wb') as output:
        pickle.dump(obj,output,-1)
        
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

#%%
def get_constants_dict(name=''):
    def openjson(root_dir):
        with open(root_dir + name + ".json", "r") as read_file:
            data = json.load(read_file)
        return data
    if name:
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__)) #directory where this script is stored.
            # Using this directory path, the module System (and the others) can be imported
            # from an arbitrary directory provided that the respective path is in the PYTHONPATH variable.
            return openjson(script_dir + "\\..\\constants\\")
        except FileNotFoundError:
            return openjson("./constants/")
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
    kwargs['mp'] = False
    for kwargs2 in kwargs['kwargs']:
        kwargs[kwargs2] = kwargs['kwargs'][kwargs2]
    del kwargs['self']
    del kwargs['kwargs']
    
    #no looping through magnetic field strength or direction for rateeqs so far
    if obj.calcmethod == 'rateeqs': obj.Bfield.reset()
    
    #___expand dimensions of strength, direction, v0, r0 in order to be able to loop through them    
    if np.array(obj.Bfield.strength).ndim == 0:   strengths = [obj.Bfield.strength]
    else:                               strengths = obj.Bfield.strength
    if np.array(obj.Bfield.direction).ndim == 1:  directions = [obj.Bfield.direction]
    else:                               directions = obj.Bfield.direction    
    if obj.r0.ndim == 1:    r0_arr = obj.r0[None,:]
    else:                   r0_arr = obj.r0
    if obj.v0.ndim == 1:    v0_arr = obj.v0[None,:]
    else:                   v0_arr = obj.v0
    
    #___loop through laser objects to get to know which variables have to get
    #___iterated and how many iterations
    #--> for the dictionaries used here it'S important that the order is ensured
    #    (this is the case since python 3.6 - now (3.8))
    laser_list = []
    laser_iters_N = {}
    for l1,la in enumerate(obj.lasers):
        laser_dict = {}
        for key in ['omega','freq_Rabi','I','phi','beta','k','r_k','f_q']:
            value = la.__dict__[key]
            if (np.array(value).ndim == 1 and key not in ['k','r_k','f_q']) \
                or (np.array(value).ndim == 2 and key in ['k','r_k','f_q']): #or also with dict comprehension
                laser_dict[key] = value
                laser_iters_N[key] = len(value)
        laser_list.append(laser_dict)
    laser_iters = list(laser_iters_N.keys())
    # if kwargs['verbose']: print(laser_list,laser_iters,laser_iters_N)
    
    #___recursive function to loop through all iterable laser variables
    def recursive(_laser_iters,index):
        if not _laser_iters:
            for i,dic in enumerate(laser_list):
                for key,value in dic.items():
                    # if kwargs['verbose']: print('Laser {}: key {} is set to {}'.format(i,key,value[index[key]]))
                    obj.lasers[i].__dict__[key] = value[index[key]]
                    #or more general here: __setattr__(self, attr_name, value)
            # if kwargs['verbose']: print('b1={},b2={},b3={},b4={}'.format(b1,b2,b3,b4))
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
    result_objects = []
    iters_dict = {'strength': len(strengths),
                  'direction': len(directions),
                  'r0':len(r0_arr),
                  'v0':len(v0_arr),
                  **laser_iters_N}
    #if v0_arr and r0_arr have the same length they should be varied at the same time and not all combinations should be calculated.
    if len(r0_arr) == len(v0_arr) and len(r0_arr) > 1: del iters_dict['v0']
    #___looping through all iterable parameters of system and laser
    for b1,strength in enumerate(strengths):
        for b2,direction in enumerate(directions):
            obj.Bfield.turnon(strength,direction)
            for b3,r0 in enumerate(r0_arr):
                obj.r0 = r0
                for b4,v0 in enumerate(v0_arr):
                    if (len(r0_arr) == len(v0_arr)) and (b3 != b4): continue
                    obj.v0 = v0
                    recursive(laser_iters,{})
                    
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
    iters_dict = {key:value for key,value in list(iters_dict.items()) if value != 1}
    for i,key in enumerate(keys):
        first_el = np.array(results[0][i])
        if first_el.size == 1:
            out[key] = np.squeeze(np.reshape(np.concatenate(
                np.array(results,dtype=object)[:,i], axis=None), tuple(iters_dict.values())))
        else:
            out[key] = np.squeeze(np.reshape(np.concatenate(
                np.array(results,dtype=object)[:,i], axis=None), tuple([*iters_dict.values(),*(first_el.shape)])))
    
    return out, iters_dict# also here iters_dict with actual values???

                    # index = {}
                    # for l1 in range(laser_iters_N['omega']):
                    #     index['omega'] = l1  
                    #     for l2 in range(laser_iters_N['k']):
                    #         index['k'] = l2
                    #         for i,dic in enumerate(laser_list):
                    #             for key,value in dic.items():
                    #                 print('Laser {}: key {} is set to {}'.format(i,key,value[index[key]]))
                    #                 # obj.lasers[i].__dict__[key] = value[index[key]]
                    # if calling_fun == 'calc_OBEs':
                        # result_objects.append(pool.apply_async(np.sum,args=(np.arange(3),)))
                        # result_objects.append(mp_calc(obj,betaB[c1,c2,:],**kwargs)) # --> without Pool parallelization
                    # result_objects.append(pool.apply_async(deepcopy(obj).calc_OBEs,kwds=(kwargs)))
#%%
def vtoT(v,mass=157):
    """function to convert a velocity v in m/s to a temperatur in K."""
    from scipy.constants import k, u
    return v**2 * 0.5*(mass*u)/k
def Ttov(T,mass=157):
    """function to convert a temperatur in K to a velocity v in m/s."""
    from scipy.constants import k, u
    return np.sqrt(k*T*2/(mass*u))