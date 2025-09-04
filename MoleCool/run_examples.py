# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 14:34:02 2025

@author: fkogel

tested with v3.4.3
"""
from pathlib import Path
from argparse import ArgumentParser
import runpy
import matplotlib.pyplot as plt

def base_dir():
    try:
        base_dir = Path(__file__).resolve().parent
    except NameError:
        # __file__ is not defined (e.g. running in IPython / Jupyter)
        base_dir = Path.cwd()
        
    return base_dir

def Examples_folder():
    return base_dir() / "Examples"

def doc_folder():
    return base_dir().parent / "doc/source"

def only_names(files): 
    return [f.name for f in files]

def run_example_scripts(filenames, args, verbose=True):    
    if args.out:
        output_dir = Path(args.out)
        output_dir.mkdir(exist_ok=True)
    
    # Loop through all python files in the directory
    for i,filename in enumerate(filenames):
        if verbose:
            n=60
            print('{s:{c}^{n}}'.format(s='',n=n,c='='))
            print(f"   {i}: executing file {filename.name}...")
            print('{s:{c}^{n}}'.format(s='',n=n,c='='))
            
        script = Path(Examples_folder()) / filename
        # Run the script in an isolated namespace
        runpy.run_path(script, run_name="__main__")
    
        if args.out:
            # Collect all open figures
            figs = [plt.figure(i) for i in plt.get_fignums()]
        
            # Save each figure
            for j, fig in enumerate(figs, start=1):
                outpath = output_dir / (f"{filename.with_suffix('')}_fig{j}." + args.type)
                outpath.parent.mkdir(exist_ok=True)
                fig.savefig(outpath)
                if verbose:
                    print(f"  Saved {outpath}")
        
            # Close all figures before moving to next script
            plt.close("all")
        
        else:
            plt.show()
    
def main():
    fnames_all      = [p.relative_to(Examples_folder())
                       for p in Examples_folder().rglob("*.py")]
    fnames_long     = [f for f in fnames_all if not f.name.startswith("plot_")]
    fnames_fast     = [f for f in fnames_all if f.name.startswith("plot_")]
    
    parser = ArgumentParser(
        prog="MoleCool_examples",
        description="Runs examples from the Examples folder.",
    )

    parser.add_argument(
        "--name", type=str, default="all",
        help=f"name of the example to be executed or 'all'.\n\
            fast examples are: {only_names(fnames_fast)}\n\
            long examples are: {only_names(fnames_long)}"
    )

    parser.add_argument(
        "--out", type=str, default='',
        help="directory where all matplotlib figures are captured and saved.",
        )
    
    parser.add_argument(
        "--type", type=str, default='png',
        help="image type of the matplotlib figures that are being saved when '--out' is provided",
        )
    
    
    # Get each argument from the command line
    args = parser.parse_args()
    
    ###################
    # Pre-generate outputs
    
    if args.name == 'all':
        run_example_scripts(fnames_all, args)
    elif args.name == 'fast':
        run_example_scripts(fnames_fast, args)
    elif args.name == 'long':
        run_example_scripts(fnames_long, args)
    else:
        match = next((p for p in fnames_all if p.name == args.name), None)
        
        if not match:
            raise ValueError(f'{args.name} not valid example name!')
        
        run_example_scripts([match], args)

if __name__ == '__main__':
    main()