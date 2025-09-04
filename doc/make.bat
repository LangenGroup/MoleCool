@echo off

D:
cd D:\git\MolLaserCoolSimulations\doc\source
REM Delete folders if they exist
IF EXIST build rmdir /s /q build
IF EXIST generated rmdir /s /q generated
IF EXIST auto_examples rmdir /s /q auto_examples
IF EXIST gen_modules rmdir /s /q gen_modules
del "sg_execution_times.rst" 2>nul

REM Activate Python environment
D:
CALL D:\activate_pyMoleCool2.bat

REM Run Sphinx build
D:
cd D:\git\MolLaserCoolSimulations\doc
sphinx-build -b html source source\build
sphinx-build -b html source source\build

cmd /k

REM PAUSE
