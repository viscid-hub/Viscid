#!/usr/bin/env python
# example of making a time series plot

from __future__ import print_function, division
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

import viscid

## The following two lines change how the file readers work...
## they can also be made the defaults by putting the following
## into a file called ~/.viscidrc:
##     readers.openggcm.GGCMFile.read_log_file: true
##     readers.openggcm.GGCMGrid.mhd_to_gse_on_read: true
# this will populate the file's info dictionary with
# some of the parameters in the log file peamble...
# if you have the log file accessable
viscid.readers.openggcm.GGCMFile.read_log_file = True
# this will automatically convert to GSE coordinates
viscid.readers.openggcm.GGCMGrid.mhd_to_gse_on_read = True

# load a whole time series as a single "file"
f = viscid.load_file("combined_txsmall/target/*.3df.*")

# iter_times only understands seconds or indices, so we need to parse the
# UT ourselves if that's how we want to slice time...
try:
    time_fmt = "%Y:%m:%d:%H:%M:%S.%f"
    dipole_time = datetime.strptime(f.info["ggcm_dipole_dipoltime"], time_fmt)
    # t0 and t1 are datetime.timedelta objects
    t0 = datetime.strptime("1967:01:01:00:00:00.00", time_fmt) - dipole_time
    t1 = datetime.strptime("1967:01:01:00:00:02.00", time_fmt) - dipole_time
    time_slice = "{0}:{1}".format(t0.total_seconds(), t1.total_seconds())
except KeyError:
    # this happens if the file read is an "MHD-in-a-box" run, or if the
    # logfile didn't read correctly... in that case, just use all times
    time_slice = ":"

times = np.array([grid.time for grid in f.iter_times(time_slice)])
bz_ts = np.empty_like(times)

for i, grid in enumerate(f.iter_times(time_slice)):
    # notice that the slice here includes .0, if the numbers
    # look like integers, the slice will be by index in the
    # array, which is sometimes desired.
    bz_ts[i] = grid['vz']["x=-10.0,y=0.0,z=0.0"]

plt.plot(times, bz_ts)
plt.xlabel("Time [s]")
plt.ylabel("$V_z$ [nT]")
plt.title(f.info['run'])
plt.show()

##
## EOF
##
