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
import time
import traceback

# ANSI color codes
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

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
    
    summary = []  # store results for each script

    # Loop through all python files in the directory
    for i, filename in enumerate(filenames):
        start_time = time.time()
        success = True
        n = 60
        
        if verbose:    
            print('{s:{c}^{n}}'.format(s='', n=n, c='='))
            print(f"   {i}: executing file {filename.name}...")
            print('{s:{c}^{n}}'.format(s='', n=n, c='='))
    
        script = Path(Examples_folder()) / filename
    
        try:
            # Run the script in an isolated namespace
            runpy.run_path(script, run_name="__main__")
        except Exception:
            success = False
            print(f"  {RED}Script {filename.name} failed with error:{RESET}")
            traceback.print_exc()   # <-- prints the full traceback
            print(RED + "  " + "-" * (n-2) + RESET)
    
        duration = time.time() - start_time
        summary.append((filename.name, success, duration))
    
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
    
        if args.show:
            plt.show()
        
        # Close all figures before moving to next script
        plt.close('all')

    # --- Print summary ---
    max_len = max(len(fname) for fname, _, _ in summary)

    print("\n" + "="*n)
    print("TEST SCRIPT SUMMARY")
    print("="*n)
    
    for fname, success, duration in summary:
        status = f"{GREEN}Success{RESET}" if success else f"{RED}Failed{RESET}"
        # Align filename left, status centered, duration right
        print(f"{fname:<{max_len}}  {status:^12}  {duration:7.2f}s")
    
    print("="*n)
    
def main():
    files   = [p.relative_to(Examples_folder()) 
               for p in Examples_folder().rglob("*.py")]
    fnames  = dict(
        all     = files.copy(),
        long    = [f for f in files if not f.name.startswith("plot_")],
        fast    = [f for f in files if f.name.startswith("plot_")],
        )
    
    parser = ArgumentParser(
        prog="MoleCool_examples",
        description="Runs examples from the Examples folder.",
    )

    parser.add_argument(
        "--name", type=str, default="fast",
        help=(f"name of the example to be executed or one of {list(fnames.keys())}.\n"
              f"fast examples are: {only_names(fnames['fast'])}\n"
              f"long examples are: {only_names(fnames['long'])}")
    )

    parser.add_argument(
        "--out", type=str, default='',
        help="directory where all matplotlib figures are captured and saved.",
        )
    
    parser.add_argument(
        "--type", type=str, default='png',
        help="image type of the matplotlib figures that are being saved when '--out' is provided",
        )
    
    parser.add_argument(
        '--show', action='store_true',
        help="Whether to enable showing all plot figures.",
        )
    
    # Get each argument from the command line
    args = parser.parse_args()
    
    ###################
    # Pre-generate outputs
    
    if args.name in fnames.keys():
        run_example_scripts(fnames[args.name], args)
    else:
        match = next((p for p in files if p.name == args.name), None)
        
        if not match:
            raise ValueError(f'{args.name} not valid example name!')
        
        run_example_scripts([match], args)

if __name__ == '__main__':
    main()