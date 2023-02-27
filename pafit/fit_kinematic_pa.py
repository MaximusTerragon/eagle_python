#!/usr/bin/env python
"""
#############################################################################

Copyright (C) 2005-2018, Michele Cappellari
E-mail: michele.cappellari_at_physics.ox.ac.uk

Updated versions of the software are available from my web page
http://purl.org/cappellari/software

If you have found this software useful for your research,
we would appreciate an acknowledgement to use of the
`Method described in Appendix C of Krajnovic et al. (2006)'.
http://adsabs.harvard.edu/abs/2006MNRAS.366..787K

This software is provided as is without any warranty whatsoever.
Permission to use, for non-commercial purposes is granted.
Permission to modify for personal or internal use is granted,
provided this copyright and disclaimer are included unchanged
at the beginning of the file. All other rights are reserved.
In particular, redistribution of the code is not allowed.

#############################################################################
+
 NAME:
   FIT_KINEMATIC_PA

 PURPOSE:
   Determine the global kinematic position angle of a
   galaxy with the method described in Appendix C of
   Krajnovic, Cappellari, de Zeeuw, & Copin 2006, MNRAS, 366, 787
   
 *** The documentation has not been updated for Python ***

 CALLING SEQUENCE:
   FIT_KINEMATIC_PA, Xbin, Ybin, Vel, AngleBest, AngleError, VelSyst, $
        /DEBUG, NSTEPS=nsteps, /QUIET

 INPUT PARAMETERS:
   XBIN, YBIN: vectors with the coordinates of the bins (or pixels)
       measured from the centre of rotation (typically the galaxy centre).
     - IMPORTANT: The routine will not give meaningful output unless
       (X,Y)=(0,0) is an estimate of the centre of rotation.
   VEL: measured velocity at the position (XBIN,YBIN).
     - IMPORTANT: An estimate of the systemic velocity has to be already
       subtracted from this velocity [e.g. VEL_CORR = VEL - np.median(VEL)].
       The routine will then provide in the output VELSYST a correction
       to be added to the adopted systemic velocity.

 INPUT KEYWORDS:
   NSTEPS: number of steps along which the angle is sampled.
       Default is 361 steps which gives a 0.5 degr accuracy.
       Decrease this number to limit computation time during testing.

 OUTPUT PARAMETER:
   ANGLEBEST: kinematical PA. Note that this is the angle along which
       |Vel| is maximum (note modulus!). If one reverses the sense of
       rotation in a galaxy ANGLEBEST does not change. The sense of
       rotation can be trivially determined by looking at the map of Vel.
   ANGLEERROR: corresponding error to assign to ANGLEBEST.
   VELSYST: Best-fitting correction to the adopted systemic velocity
       for the galaxy.
     - If the median was subtracted to the input velocity VEL before
       the PA fit, then the corrected systemnic velocity will be
       np.median(VEL) + VELSYST.
    - EXAMPLE:

 MODIFICATION HISTORY:
   V1.0.0 -- Created by Michele Cappellari, Leiden, 30 May 2005
   V1.1.0 -- Written documentation. MC, Oxford, 9 October 2007
   V1.1.1 -- Corrected handling of NSTEPS keyword. Thanks to Roland Jesseit.
       MC, Oxford, 17 October 2007
   V1.1.2 -- Force error to be larger than 1/2 of the angular step size.
       MC, Oxford, 19 October 2007
   V1.1.3 -- Determine plotting ranges from velSym instead of vel.
       Thanks to Anne-Marie Weijmans. Leiden, 25 May 2008
   V1.1.4 -- Clarified that systemic velocity has to be subtracted from VEL.
       MC, Oxford, 31 March 2009
   V1.1.5 -- Overplot best PA on data. Some changes to the documentation.
       MC, Oxford, 14 October 2009
   V1.2.0 -- Includes error in chi^2 in the determination of angErr.
       Thanks to Davor Krajnovic for reporting problems.
       MC, Oxford, 23 March 2010
   V1.3.0 -- The program is two orders of magnitude faster, thanks to a
       new cap_symmetrize_velfield routine. MC, Oxford, 8 October 2013
   V1.3.1 -- Uses CAP_RANGE routine to avoid potential naming conflicts.
       Uses TOLERANCE keyword of TRIANGULATE to try to avoid IDL error
       "TRIANGULATE: Points are co-linear, no solution."
       MC, Oxford, 2 December 2013
   V2.0.0 -- Translated from IDL into Python. MC, Oxford, 10 April 2014
   V2.0.1 -- Support both legacy Python 2.7 and Python 3. MC, Oxford, 25 May 2014
   V2.0.2 -- Fixed possible program stop. MC, Sydney, 2 February 2015
   V2.0.3 -- Changed color of plotted lines. Check matching of input sizes.
       MC, Oxford, 7 September 2015
   V2.0.4 -- Use np.percentile instead of deprecated Scipy version.
       MC, Oxford, 17 March 2017
   V2.0.5: Changed imports for plotbin as package. MC, Oxford, 17 April 2018
   V2.0.6: Dropped Python 2.7 support. MC, Oxford 12 May 2018

"""

import numpy as np
import matplotlib.pyplot as plt
import collections.abc

from plotbin.symmetrize_velfield import symmetrize_velfield
from plotbin.plot_velfield import plot_velfield

##############################################################################

def fit_kinematic_pa(x, y, vel, debug=False, nsteps=361,
                     quiet=False, plot=True, dvel=10):

    x, y, vel = map(np.ravel, [x, y, vel])

    assert x.size == y.size == vel.size, 'Input vectors (x, y, vel) must have the same size'


    #print('pafit', dvel)


    if isinstance(dvel, collections.abc.Sequence) == True:
        print('CHECK SETTINGS', dvel)
        assert x.size == y.size == vel.size == dvel.size, 'Input vectors (x, y, vel) must have the same size'
        

    nbins = x.size
    n = nsteps
    angles = np.linspace(0, 180, n) # 0.5 degrees steps by default
    chi2 = np.empty_like(angles)
    for j, ang in enumerate(angles):
        velSym = symmetrize_velfield(x, y, vel, sym=1, pa=ang)
        #reduced chi squared
        chi2[j] = np.sum(((vel - velSym)/dvel)**2)
        if debug:
            print('Ang: %5.1f, chi2/DOF: %.4g' % (ang, chi2[j]/nbins))
            plt.cla()
            plot_velfield(x, y, velSym)
            plt.pause(0.01)
    k = np.argmin(chi2)
    angBest = angles[k]

    # Compute fit at the best position
    #
    velSym = symmetrize_velfield(x, y, vel, sym=1, pa=angBest)
    
    #-----------------------------
    # Start of Max modifications:
    
    #print('angBest', angBest)
    
    # Will find angle from y-axis to negative
    ang = [0,np.pi] + np.radians(angBest)
    mask = np.where(((np.arctan2(y,x) + np.pi) > ang[0]) & ((np.arctan2(y,x) + np.pi) < ang[1]))
    
    #print('white mean: ', np.mean(velSym[mask]))
    
    mask_other = np.array([i for i in np.arange(0, len(velSym), 1) if i not in mask[0]])
    #print('black other: ', np.mean(velSym[mask_other]))
    
    #if (np.mean(velSym[mask]) < 0) & (np.mean(velSym[mask_other]) < 0):
    #    print('velSym double negative')
    #    if np.mean(velSym[mask]) > np.mean(velSym[mask_other]):
    #       print('velSym FLIPPED as greater negative exists')
    #if (np.mean(velSym[mask]) > 0) & (np.mean(velSym[mask_other]) > 0):
        #print('velSym double positive')
        #if np.mean(velSym[mask]) < np.mean(velSym[mask_other]):
            #print('velSym FLIPPED as greater positive exists')
	    	
	
    """WE WANT MASK TO BE THE LARGER ONE
    - -- ... FAILS... both negative, so assumes - is negative and -- is positive... so we flip it if mask > mask_other
    -- - ... TRUE.... both negative, and correctly assumes -- is negative, so we dont flip if mask < mask_other
    + ++ ... FAILS... both positive, so assumes + is positive and ++ is negative... so we clearly flip if mask < mask_other
    ++ + ... TRUE... both positive, and correctly assymes ++ is positive, so we dont flip if mask > mask_other"""
    
    if np.mean(velSym[mask]) < np.mean(velSym[mask_other]):
        angBest += 180
        if angBest > 360:
            angBest = angBest - 360 
        #print('angBest mean flipped:', angBest)
            
    x_mask = x[mask]
    y_mask = y[mask]
    
    #-----------------------------
    
    # 3sigma confidence limit, including error on chi^2
    #
    f = chi2 - chi2[k] <= 9 + 3*np.sqrt(2*nbins)
    minErr = max(0.5, (angles[1] - angles[0])/2.0)
    if f.sum() > 1:
        angErr = (np.max(angles[f]) - np.min(angles[f]))/2.0
        if angErr >= 45:
            good = np.degrees(np.arctan(np.tan(np.radians(angles[f]))))
            angErr = (np.max(good) - np.min(good))/2.0
        angErr = max(angErr, minErr)
    else:
        angErr = minErr

    vSyst = np.median(vel - velSym)

    if not quiet:
        print('  Kin PA: %5.1f' % angBest, ' +/- %5.1f' % angErr, ' (3*sigma error)')
        print('Velocity Offset: %.2f' % vSyst)

    # Plot results
    #
    if plot:

        mn, mx = np.percentile(velSym, [2.5, 97.5])
        mx = min(mx, -mn)
        plt.subplot(121)
        plot_velfield(x, y, velSym, vmin=-mx, vmax=mx)
        plt.title('Symmetrized')

        plt.subplot(122)
        plot_velfield(x, y, vel - vSyst, vmin=-mx, vmax=mx)
        plt.title('Data and best PA')
        rad = np.sqrt(np.max(x**2 + y**2))
        ang = [0,np.pi] + np.radians(angBest)
        plt.plot(rad*np.cos(ang), rad*np.sin(ang), 'k--', linewidth=2) # Zero-velocity line
        plt.plot(-rad*np.sin(ang), rad*np.cos(ang), color="limegreen", linewidth=3) # Major axis PA
        
        # Max change
        #mask = np.where(((np.arctan2(y,x) + np.pi) > ang[0]) & ((np.arctan2(y,x) + np.pi) < ang[1]))
        #x = x[mask]
        #y = y[mask]
        plt.scatter(x_mask, y_mask, c='w', s=0.5, zorder=100)
        plt.arrow(0, 0, 0.4*rad*np.cos(ang[0]), 0.4*rad*np.sin(ang[0]), zorder=101, color='k', width=0, head_width=2, head_length=2)

    return angBest, angErr, vSyst

##############################################################################

# Usage example for fit_kinematic_pa()

if __name__ == '__main__':

    xbin, ybin = np.random.uniform(low=[-30, -20], high=[30, 20], size=(100, 2)).T
    inc = 60.                       # assumed galaxy inclination
    r = np.sqrt(xbin**2 + (ybin/np.cos(np.radians(inc)))**2) # Radius in the plane of the disk
    a = 40                          # Scale length in arcsec
    vr = 2000*np.sqrt(r)/(r + a)      # Assumed velocity profile
    vel = vr * np.sin(np.radians(inc))*xbin/r # Projected velocity field

    plt.clf()
    fit_kinematic_pa(xbin, ybin, vel, debug=True, nsteps=30)
    plt.pause(1)
