Changelog
---------

V3.1.5: MC, Oxford, 24 February 2022
++++++++++++++++++++++++++++++++++++

- Included test for repetitions in ``_cvt_equal_mass`` to avoid the 
  possibility of cycling. Thanks to Peter Weilbacher (aip.de) for the 
  report, a detailed example producing the problem, and subsequent testing. 

V3.1.4: MC, Oxford, 19 February 2020
++++++++++++++++++++++++++++++++++++

- Formatted documentation as docstring. 

V3.1.3: MC, Oxford, 27 September 2018
+++++++++++++++++++++++++++++++++++++

- Fixed clock DeprecationWarning in Python 3.7.

V3.1.2: MC, Oxford, 10 May 2018
+++++++++++++++++++++++++++++++

- Dropped legacy Python 2.7 support. 

V3.1.1: MC, Oxford, 15 September 2017
+++++++++++++++++++++++++++++++++++++

- Stops if unbinned pixels do not have enough S/N.
- Removed weighted centroid function. 

V3.1.0: MC, Oxford, 17 July 2017
++++++++++++++++++++++++++++++++

- Use cKDTree for much faster un-weighted Voronoi Tessellation.
- Removed loop over bins from Lloyd's algorithm with CVT.
      
V3.0.9: MC, Oxford, 10 July 2017
++++++++++++++++++++++++++++++++

- Do not iterate down to diff==0 in _cvt_equal_mass().
- Request `pixelsize` when dataset is large. Thanks to Davor Krajnovic
  (Potsdam) for the feedback. 
- Make `quiet` really quiet.
- Fixed some instances where `sn_func()` was not being used (only 
  relevant when passing the `sn_func` keyword). 

V3.0.8: MC, Oxford, 15 February 2017
++++++++++++++++++++++++++++++++++++

- New voronoi_tessellation() function. 

V3.0.7: MC, Oxford, 23 January 2017 
+++++++++++++++++++++++++++++++++++

- Print execution time. 

V3.0.6: MC, Oxford, 14 June 2016
++++++++++++++++++++++++++++++++

- Use interpolation='nearest' to avoid crash on MacOS.
  Thanks to Kyle Westfall (Portsmouth) for reporting the problem.
- Allow for zero noise. 

V3.0.5: MC, Oxford, 18 April 2016
+++++++++++++++++++++++++++++++++

- Fixed deprecation warning in Numpy 1.11. 

V3.0.4: MC, Oxford, 12 April 2016
+++++++++++++++++++++++++++++++++

- Included keyword "sn_func" to pass a function which
  calculates the S/N of a bin, rather than editing _sn_func().
- Included test to prevent the addition of a pixel from
  ever decreasing the S/N during the accretion stage.
      
V3.0.3: MC, Oxford, 31 March 2016
+++++++++++++++++++++++++++++++++

- Use for loop to calculate Voronoi tessellation of large arrays
  to reduce memory usage. Thanks to Peter Weilbacher (Potsdam) for
  reporting the problem and providing the solution.
      
V3.0.2: MC, Oxford, 2 October 2014
++++++++++++++++++++++++++++++++++

- Avoid potential runtime warning while plotting.

V3.0.1: MC, Oxford, 25 May 2014
+++++++++++++++++++++++++++++++

- Support both Python 2.7 and Python 3. 
      
V3.0.0: MC, London, 19 March 2014
+++++++++++++++++++++++++++++++++

- Translated from IDL into Python and tested against the original.
      
V2.6.0: MC, London, 19 March 2014
+++++++++++++++++++++++++++++++++

- Included new SN_FUNCTION to illustrate the fact that the user can
  define a custom function to estimate the S/N of a bin if needed.
      
V2.5.8: MC, La Palma, 15 May 2012
+++++++++++++++++++++++++++++++++

- Update Voronoi tessellation at the exit of bin2d_cvt_equal_mass.
  This is only done when using /WVT, as DIFF may not be zero at the
  last iteration. 

V2.5.7: MC, Oxford, 24 March 2012
+++++++++++++++++++++++++++++++++

- Included safety termination criterion of Lloyd algorithm
  to prevent loops using /WVT. 

V2.5.6: MC, Oxford, 11 November 2011
++++++++++++++++++++++++++++++++++++

- Use IDL intrinsic function DISTANCE_MEASURE for automatic pixelSize, 
  when PIXSIZE keyword is not given.
      
V2.5.5: MC, Oxford, 28 April 2010
+++++++++++++++++++++++++++++++++

- Added PIXSIZE keyword. 
      
V2.5.4: MC, Oxford, 30 November 2009
++++++++++++++++++++++++++++++++++++

- Improved color shuffling for final plot.

V2.5.3: MC, Oxford, 3 December 2007
+++++++++++++++++++++++++++++++++++

- Fixed program stop, introduced in V2.5.0, with /NO_CVT keyword.
      
V2.5.2: MC, Oxford, 28 March 2007
+++++++++++++++++++++++++++++++++

- Print number of unbinned pixels. 
      
V2.5.1: MC, Oxford, 3 November 2006
+++++++++++++++++++++++++++++++++++

- Updated documentation. 

V2.5.0: MC, Leiden, 9 March 2006
++++++++++++++++++++++++++++++++

- Added two new lines of code and the corresponding /WVT keyword
  to implement the nice modification to the algorithm proposed by
  Diehl & Statler (2006). 

V2.4.8: MC, Leiden, 23 December 2005
++++++++++++++++++++++++++++++++++++

- Use geometric centroid of a bin during the bin-accretion stage,
  to allow the routine to deal with negative signal (e.g. in
  background-subtracted X-ray images). Thanks to Steven Diehl for
  pointing out the usefulness of dealing with negative signal.
      
V2.4.7: MC, Leiden, 27 September 2005
+++++++++++++++++++++++++++++++++++++

- Verify that SIGNAL and NOISE are non negative vectors.
      
V2.4.6: MC, Leiden, 27 August 2005
++++++++++++++++++++++++++++++++++

- Added /NO_CVT keyword to optionally skip the CVT step of
  the algorithm. 

V2.4.5: MC, Leiden, 3 December 2004
+++++++++++++++++++++++++++++++++++

- Added BIN2D prefix to internal routines to avoid possible
  naming conflicts. 

V2.4.4: MC, Leiden, 30 November 2004
++++++++++++++++++++++++++++++++++++

- Prevent division by zero for pixels with signal=0
  and noise=sqrt(signal)=0, as can happen from X-ray data.
      
V2.4.3: MC, Leiden, 29 November 2004
++++++++++++++++++++++++++++++++++++

- Corrected bug introduced in version 2.3.1. It went undetected
  for a long time because it could only happen in special conditions.
  Now we recompute the index of the good bins after computing all
  centroids of the reassigned bins in reassign_bad_bins. Many thanks
  to Simona Ghizzardi for her clear analysis of the problem and
  the solution. 

V2.4.2: MC, Leiden, 4 August 2004
+++++++++++++++++++++++++++++++++

- Use LONARR instead of INTARR to define the CLASS vector,
  to be able to deal with big images. Thanks to Tom Statler.
      
V2.4.1: MC, Leiden, 14 December 2003
++++++++++++++++++++++++++++++++++++

- Added /QUIET keyword and verbose output during the computation.
  After suggestion by Richard McDermid. 

V2.4.0: MC, Leiden, 10 December 2003
++++++++++++++++++++++++++++++++++++

- Addedd basic error checking of input S/N. 
- Reintroduced the treatment for zero-size bins in CVT, which 
  was deleted in V2.2. Thanks to Robert Sharp and Kambiz Fathi 
  for reporting problems.

V2.3.1: MC, Leiden, 13 April 2003
+++++++++++++++++++++++++++++++++

- Do *not* assume the first bin is made of one single pixel.
- Added computation of S/N scatter and plotting of 1-pixel bins.
      
V2.3.0: MC, Leiden, 9 April 2003
++++++++++++++++++++++++++++++++

- Unified the three tests to stop the accretion of one bin.
  This can improve some bins at the border. 

V2.2.0: MC, Leiden, 11 March 2003
+++++++++++++++++++++++++++++++++

- Added computation of useful bin quantities in output. Deleted some
  safety checks for zero size bins in CVT. Minor polishing of the code.
      
V2.1.0: MC, Vicenza, 13 February 2003
+++++++++++++++++++++++++++++++++++++

- First released version. Written documentation.
      
V2.0.0: MC, Leiden, 11 September 2001
+++++++++++++++++++++++++++++++++++++

- Major revisions. Stable version. 

V1.0.0: Michele Cappellari, Leiden, June 2001
+++++++++++++++++++++++++++++++++++++++++++++

- First working implementation. 
