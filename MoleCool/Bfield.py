# -*- coding: utf-8 -*-
"""
This module contains the class :class:`Bfield` which provides methods to represent
and imitate a realistic DC magnetic field.
The main intended purpose of this class is to be part of the :class:`System.System`
in order to include the effect of a megnetic field in dynamics simulations.
"""
import numpy as np
from scipy.constants import c,h,hbar,pi,g,physical_constants
from scipy.constants import k as k_B
from scipy.constants import u as u_mass
import warnings
#%%
class Bfield:
    def __init__(self,**kwargs):
        """Class defines a magnetic field configuration and methods to turn on
        a certain field strength and direction conveniently. When initializing
        a new system via `system=System()` a magnetic field instance with zero
        field strength is directly included in this System via `system.Bfield`.
        
        Example
        -------
        >>> B1 = Bfield()   # initialize Bfield instance
        >>> B1.turnon(strength=5e-4,direction=[0,0,1],angle=60)
        >>> print(B1)       # print properties
        >>> B1.reset()      # reset magnetic field to zero.

        Parameters
        ----------
        **kwargs
            Optional keyword arguments for directly turn on a certain magnetic
            field by using these keyword arguments within the the method
            :func:`turnon` (further information).
        """
        if kwargs:  self.turnon(**kwargs)
        else:       self.reset()
        self.mu_B = physical_constants['Bohr magneton'][0]
        
    def turnon(self,strength=5e-4,direction=[0,0,1],angle=None,remix_strength=None):
        """Turn on a magnetic field with a certain strength and direction.

        Parameters
        ----------
        strength : float or np.ndarray, optional
            Strength in Tesla. The default is 5e-4.
        direction : list or np.ndarray with shape (3,), optional
            Direction of the magnetic field vector. Doesn't have to be given
            as normalized array. The default is [0,0,1].
        angle : float or np.ndarray, optional
            Angle in degrees at which the magnetic field vector is pointing with
            respect to the `direction` argument. The default is None.
        remix_strength : float, optional
            measure of the magnetic field strength (i.e. the magnetic remixing
            matrix is multiplied by 10^remix_strength). Reasonable values are
            between 6 and 9. The default is None.
        """
        if np.any(strength >= 10e-4):
            print('WARNING: linear Zeeman shifts are only a good approx for B<10G.')
        self.strength = strength
        self.direction = np.array(direction) / np.expand_dims(np.linalg.norm(direction,axis=-1),axis=-1)
        if np.all(angle != None):
            self.angle = angle
            self.axisforangle = self.direction
            angle = angle/360*2*pi
            v1      = self.direction
            v_perp  = np.cross(v1,np.array([0,1,0]))
            if np.all(v_perp == 0.0):
                v_perp = np.cross(v1, np.array([1,0,0]))
            self.direction = np.tensordot(np.cos(angle), v1, axes=0) \
                             + np.tensordot(np.sin(angle), v_perp, axes=0)
        
    def turnon_earth(self,vertical='z',towardsNorthPole='x'):
        """Turn on the magnetic field of the earth at Germany with a strength
        of approximately 48 uT. The vertical component is 44 uT and the horizontal
        component directing towards the North Pole is 20 uT.

        Parameters
        ----------
        vertical : str, optional
            vertical axis. Supported values are 'x', 'y' or 'z'.
            The default is 'z'.
        towardsNorthPole : str, optional
            horizontal axis directing towards the North Pole. Supported values
            are 'x', 'y' or 'z'.The default is 'x'.
        """
        axes = {'x' : 0, 'y' : 1, 'z' : 2}
        vec = np.zeros(3)
        vec[axes[vertical]]         = 44e-6
        vec[axes[towardsNorthPole]] = 20e-6
        self.turnon(strength=np.linalg.norm(vec),direction=vec)
        
    def reset(self):
        """Reset the magnetic field to default which is a magnetic field
        strength 0.0 and the direction [0.,0.,1.]"""
        self.strength, self.direction = 0.0, np.array([0.,0.,1.])
        if 'angle' in self.__dict__: del self.angle, self.axisforangle
        self._remix_matrix = np.array([[],[]])
        
    def get_remix_matrix(self,grs,remix_strength=None):
        """return a matrix to remix all adjacent ground hyperfine levels
        by a magnetic field with certain field strength. The default is False.
    
        Parameters
        ----------
        grs : :class:`~Levelsystem.ElectronicGrState`
            for which the matrix is to be build.
        remix_strength : float
            measure of the magnetic field strength (i.e. the magnetic remixing
            matrix is multiplied by 10^remix_strength). Reasonable values are
            between 6 and 9.
        
        Returns
        -------
        array
            magnetic remixing matrix.
        """
        #must be updated!? not only grstates can mix!
        # maybe use OBEs muMat for Bfield strength
        matr = np.zeros((grs.N,grs.N))
        for i in range(grs.N):
            for j in range(grs.N):
                if grs[i].is_lossstate or grs[j].is_lossstate:
                    if grs[i].is_lossstate == grs[j].is_lossstate:
                        matr[i,j] = 1
                elif grs[i].is_equal_without_mF(grs[j]) and abs(grs[i].mF-grs[j].mF) <= 1:
                    matr[i,j] = 1
        self._remix_matrix = 10**(remix_strength)*matr
        return self._remix_matrix #if remix_strength ==None: estimate it with strength & if strength=0 return empty matrix?
    
    def __str__(self):
        return str(self.__dict__)
    
    def Bfield_vec(self):
        """Returns the magnetic field vector with its three components in T.

        Returns
        -------
        np.ndarray(3)
            magnetic field vector.
        """
        return np.tensordot(self.strength,self.direction,axes=0)
    
    @property
    def Bvec_sphbasis(self):
        """returns the magnetic field vector in the spherical basis."""
        strength, direction = self.strength, np.array(self.direction)
        ex,ey,ez = direction.T / np.linalg.norm(direction,axis=-1)
        eps = np.array([+(ex - 1j*ey)/np.sqrt(2), ez, -(ex + 1j*ey)/np.sqrt(2)])
        eps = np.array([ -eps[2], +eps[1], -eps[0] ])
        if type(strength)   == np.ndarray: strength = strength[:,None,None]
        if type(ex)         == np.ndarray: eps = (eps.T)[None,:]
        self._Bvec_sphbasis = eps*strength
        return self._Bvec_sphbasis
    
if __name__ == '__main__':
    B1 = Bfield()   # initialize Bfield instance
    B1.turnon(strength=5e-4,direction=[0,0,1],angle=60)
    print(B1)       # print properties
    B1.reset()      # reset magnetic field to zero.