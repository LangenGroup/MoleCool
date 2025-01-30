# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 14:34:02 2025

@author: fkogel

tested with v3.4.3
"""
import os, glob
from .Examples import *
from argparse import ArgumentParser

fnames_fast = [
    "RabiOsci_2level.py",
    "Simple3+1.py",
    "SimpleTest1_BaF.py",
    "SimpleTest2Traj_BaF.py",
    "EIT.py",
    "Simple3+1_AppC2.py",
    # "EIT_ims.py",
    ]

def Examples_folder():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Examples')

def run_example_scripts(filenames, verbose=True):
    for i, filename in enumerate(filenames):
        if verbose:
            n=60
            print('{s:{c}^{n}}'.format(s='',n=n,c='='))
            print(f"   {i}: executing file {filename}...")
            print('{s:{c}^{n}}'.format(s='',n=n,c='='))
        filepath = os.path.join(Examples_folder(), filename)
        os.system(f"python {filepath}")
    
def main():
    fnames_all = [os.path.basename(fname)
                      for fname in glob.glob(os.path.join(Examples_folder(), "*.py"))]
    
    parser = ArgumentParser(
        prog="MoleCool_examples",
        description="Runs examples from the Examples folder.",
    )

    parser.add_argument(
        "name", type=str, default="all",
        help=f"name of the example file to be executed or 'all'. Example files are: {fnames_all}",
    )

    # Get each argument from the command line
    args = parser.parse_args()
    
    if args.name == 'all':
        run_example_scripts(fnames_all)
    elif args.name == 'fast':
        run_example_scripts(fnames_fast)
    else:
        if args.name not in fnames_all:
            raise ValueError(f'{args.name} not valid example name!')
        run_example_scripts([args.name])

if __name__ == '__main__':
    main()