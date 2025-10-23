# rjb_phd_paper1_code
Code directly relating to my 1st PhD paper.

Stored elsewhere:
 - ERA5 and AWI Sea Ice Portal data
 - Mooring time series data
 - Model code and output (cloned from the official MITgcm repo in late 2024)

The sea ice data was accessed from AWI's HTTPS server at https://data.meereisportal.de/data/iup/hdf/s/. If this doesn't work, it can also be accessed from the IUP's server at https://data.seaice.uni-bremen.de/amsr2/. 

ERA5 data was accessed using the CDS API client Python package. See `cds_request.py` for example usage.

The mooring time series 
