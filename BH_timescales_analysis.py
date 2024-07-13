



#BH code:

#We want a way to overlay the BH growths of misaligned galaxies, over that of non-misaligned and counter-rotators

# CREATE BH TREE:
#- import galaxy tree
#- import misalignment tree
#	- extract selection criteria
#	- Option to include/exclude/any mergers
#	- Needs minimum trelax to consider (0.5 Gyr?)
#	- Set window to consider (1-2 Gyr?)
#	- Additional inputs of:
#		- Average sSFR, SFR, stellar mass, BH mass, kappa, kappa_pre/post
#	- Go through misalignment tree, find all misalignments meeting min trelax and other inputs
#	- 
#	- Extract sample from misalignment tree, extract all BH values to a misaligned_BH_dict + average values
#- start search in galaxy tree for galaxies that:
#	- meet basic selection criteria over min trelax
#	- are NOT misaligned / are COUNTER ROTATING for entire period
#	- meet average stellar mass, SFR, sSFR, BH mass, Kappa, kappa_pre/post, as single value
#	- Check mergers
#	- Extract all BH values to aligned_BH_dict and counter_BH_dict + average values
#- Create a new tree to save, as BH_tree{'misaligned': {}, 'aligned': {}, 'counter': {}} + inputs

## ANALYSE TREE:
#- import BH_tree
#- Should now have a sample of misaligned, aligned, and counter-rotating galaxies with all BH growth values
#	- Overlay all BH growths
#	- Trim to minimum trelax, so that even a 1 Gyr misalignment we only have the first 0.5 Gyr of it
#	- plot medians
#	- Randomly select X for control to plot
#	- Do we see a difference in value?
	

#do we need a max trelax too?
#issue being that misalignment_tree will give us different lengths of time - some only 0.5 Gyr, some 1 Gyr
#taking from misalignment_tree.... extract based on min trelax, but then keep extracting beyond that to a minimum of 1 gyr
#alternatively... see final line of analyse tree
#... depends a lot on how big our misalignment sample is that we get to work with here