# rjb_phd_paper1_code
Code directly relating to my 1st PhD paper.

---------------------------------
## Directory contents
 - mooring_atm_and_ice_analyses.py : Analysis and plotting of atmospheric and sea ice conditions at the mooring
 - mooring_time_series_analyses.py : Analysis and plotting of time series data from the mooring itself
 - cds_request.py : For accessing Copernicus (ERA5) data
 - More to come!

---------------------------------
## Data overview 

All datasets are all stored elsewhere in other directories. These include:
 - ERA5 and AWI Sea Ice Portal data;
 - Mooring time series data; and
 - Model code and output (cloned from the official MITgcm repo in late 2024).

Data history:
 - Sea ice data was accessed from AWI's HTTPS server at https://data.meereisportal.de/data/iup/hdf/s/. If this doesn't work, it can also be accessed from the IUP's server at https://data.seaice.uni-bremen.de/amsr2/. 
 - ERA5 data was accessed using the CDS API client Python package. See `cds_request.py` for example usage.
 - The mooring time series was provided by Markus Janout.
