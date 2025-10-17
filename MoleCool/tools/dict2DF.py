# -*- coding: utf-8 -*-
"""
This module contains several functions as tools for converting the .json files
with all the specific constants of a certain atom, molecule into pandas.DataFrames.
These DataFrames will be imported in the class :class:`~.Levelsystem.Levelsystem`
where one can have a look in a nice representation and also further
customize them in a simple way.

So, to nicely print all properties and constants for e.g. 138BaF try::
    
    levels = Levelsystem(load_constants='138BaF')
    levels.add_all_levels(0)
    levels.print_properties()
"""
import numpy as np
from scipy.constants import c,h,hbar,pi,g,u
import pandas as pd
from collections.abc import Iterable
import os
from copy import deepcopy
#%%

def get_keylist(name,dic):
    for key,value in dic.items():
        if key == name:
            return key
        elif isinstance(value, dict):
            out = get_keylist(name,value)
            if out != False:
                if out == name:
                    return [key,out]
                else:
                    return [key,*out]
    return False

def del_item(keylist,dic):
    if len(keylist) == 1:
        dic.pop(keylist[0])
    else:
        del_item(keylist[1:],dic[keylist[0]])
    
def get_value(keylist,dic):
    if len(keylist) == 1:
        return dic[keylist[0]]
    else:
        return get_value(keylist[1:],dic[keylist[0]])

def split_key(key):
    if "<-" in key:
        key1, key2 = key.replace(" ","").split("<-")
        return split_key(key1), split_key(key2)
    else:
        name, value = key.replace(" ","").split("=")
        if not (name in ['exs','gs']):
            value = float(value)
        return name,value
            
def decomp_keylist(keylist):
    if keylist[0] == 'level-specific':
        names = []
        values = []
        for i,key in enumerate(keylist[1:-1]):
            name,value = split_key(key)
            names.append(name)
            values.append(value)
        return names,values
    elif keylist[0] == 'transition-specific':
        names_row, names_col = [], []
        values_row, values_col = [], []
        for i,key in enumerate(keylist[1:-1]):
            (name_row,value_row),(name_col,value_col) = split_key(key)
            names_row.append(name_row)
            names_col.append(name_col)
            values_row.append(value_row)
            values_col.append(value_col)
        return (names_row,values_row),(names_col,values_col)
    

def get_QuNrs(keylist,dic):
    if len(keylist) == 1:
        lab,lab1,lab2=('QuNrs','QuNrs_rows','QuNrs_cols')
        if (lab1 in dic) and (lab2 in dic):
            return ([ list(dic[lab1].keys()), list(dic[lab1].values()) ],
                    [ list(dic[lab2].keys()), list(dic[lab2].values()) ])
        elif lab in dic:
            return [ list(dic[lab].keys()), list(dic[lab].values()) ]
        else:
            return None
    else:
        return get_QuNrs(keylist[1:],dic[keylist[0]])

def get_pdMultiIndex(keylist,dic):
    if keylist[0] == 'level-specific':
        names1, values1 = decomp_keylist(keylist)
        out2 = get_QuNrs(keylist, dic)
        if out2 == None:
            return pd.Index(values1, name=names1[0])#pd.MultiIndex.from_arrays(values1, names=names1)
        else:
            names2, values2 = out2
            return pd.MultiIndex.from_arrays(
                [[i]*len(values2[0]) for i in values1] + values2, names=names1+names2)
    elif keylist[0] == 'transition-specific':
        (names1_row, values1_row), (names1_col, values1_col) = decomp_keylist(keylist)
        (names2_row, values2_row), (names2_col, values2_col) = get_QuNrs(keylist, dic)
        MultiIndex_row = pd.MultiIndex.from_arrays(
            [[i]*len(values2_row[0]) for i in values1_row] + values2_row, names=names1_row+names2_row)
        MultiIndex_col = pd.MultiIndex.from_arrays(
            [[i]*len(values2_col[0]) for i in values1_col] + values2_col, names=names1_col+names2_col)
        return MultiIndex_row, MultiIndex_col
    
def get_DataFrame(dic,name,gs_exs=None,gs=None,exs=None):
    dic = deepcopy(dic)
    arr = []
    for i in range(100):
        keylist = get_keylist(name, dic)
        if keylist == False:
            break
        out = get_pdMultiIndex(keylist,dic)
        if (keylist[0] == 'level-specific') and (gs_exs in out):#not so well-programmed?
            arr.append( pd.Series(get_value(keylist, dic),
                                  index=out) )
        elif (keylist[0] == 'transition-specific') and ((gs in out[0]) and (exs in out[1])):
            arr.append( pd.DataFrame(get_value(keylist, dic),
                                     index=out[0], columns=out[1]) )
        del_item(keylist, dic)

    if len(arr) == 0:   return arr
    else:               return pd.concat(arr)#grSeries,exSeries
    
def filter_DF(DF,**QuNrs):
    keys = list(QuNrs.keys())
    key = keys[0]
    if len(keys) == 1:
        return DF.loc[DF[key] == QuNrs[key]] 
    else:
        newdic = QuNrs.copy()
        del newdic[key]
        return filter_DF(DF.loc[DF[key] == QuNrs[key]], **newdic)

def get_levels(dic,gs_exs,**QuNrs):
    dic = deepcopy(dic)
    arr = []
    for i in range(100):
        keylist = get_keylist('HFfreq', dic)
        if keylist == False:
            break # or return None
        out = get_pdMultiIndex(keylist,dic)
        if gs_exs in out:#not so well-programmed?
            arr.append( out.to_frame().reset_index(drop=True))
        del_item(keylist, dic)
    arr = pd.concat(arr)
    # if 'v' in QuNrs:
    #     if isinstance(QuNrs['v'],Iterable):
    #         arr2 = []
    #         for i,v in enumerate(QuNrs['v']):
    #             newdic = QuNrs.copy()
    #             newdic['v'] = v
    #             if i == 0:
    #                 arr2 = filter_DF(arr,**newdic)
    #             else: 
    #                 arr2 = arr.append(filter_DF(arr,**newdic))   
    if len(QuNrs) != 0:
        arr = filter_DF(arr,**QuNrs)#grSeries,exSeries
    return [arr.iloc[i].to_dict() for i in range(len(arr))]
#%%


# def is_in_dict(name,dic):
#     for key,value in dic.items():
#         if key == name:
#             return True
#         elif isinstance(value, dict):
#             if is_in_dict(name,value): return True
#     return False

# def dMat(**kwargs):
#     if not is_in_dict('dMat', dic): return None
#     pass


# def make_HFfreq(dic):
#     series_arr = []
#     # grSeries = []
#     # exSeries = []
#     for i in range(100):
#         out = get_keylist('Gamma', dic)
#         if out == False:
#             break # or return None?
#         series = pd.Series(get_value(out, dic),
#                            index=get_pdMultiIndex(out, dic))
#         series_arr.append(series)
#         # if names1[0] == 'gs': grSeries.append(series)
#         # elif names1[0] == 'exs': exSeries.append(series)
#         del_item(out, dic)
#     return series_arr#grSeries,exSeries

# def make_dMat(dic):
#     series_arr = []
#     for i in range(100):
#         out = get_keylist('vibrbranch', dic)
#         if out == False:
#             break # or return None
#         MultiIndex_row,MultiIndex_col = get_pdMultiIndex(out,dic)
#         series = pd.DataFrame(get_value(out, dic),
#                               index=MultiIndex_row, columns=MultiIndex_col)
#         series_arr.append(series)
#         # if names1[0] == 'gs': grSeries.append(series)
#         # elif names1[0] == 'exs': exSeries.append(series)
#         del_item(out, dic)
#     return series_arr#grSeries,exSeries

    