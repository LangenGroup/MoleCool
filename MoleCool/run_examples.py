# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 14:34:02 2025

@author: fkogel

tested with v3.4.0
"""
import os
from .Examples import *    

if __name__ == '__main__':
    fname = input("Example's script name: ")
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Examples', fname)
    os.system(f"python {filepath}")