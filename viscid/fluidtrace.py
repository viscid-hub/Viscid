#!/usr/bin/env python
"""Trace fluid elements in time"""

from __future__ import print_function
import itertools

import numpy as np

import viscid
from viscid.compat import izip
from viscid.calculator import calc_streamlines
from viscid.calculator import cycalc
from viscid.calculator import streamline
from viscid import logger
from viscid import seed


__all__ = ['follow_fluid', 'follow_fluid_generic']


def follow_fluid(vfile, time_slice, initial_seeds, plot_function,
                 stream_opts, **kwargs):
    """Trace fluid elements

    Note:
        you want speed_scale if say V is in km/s and x/y/z is in Re ;)

    Parameters:
        vfile: a vFile object that we can call iter_times on
        time_slice: string, slice notation, like 1200:2400:1
        initial_seeds: any SeedGen object
        plot_function: function that is called each time step,
            arguments should be exactly: (i [int], grid, v [Vector
            Field], v_lines [result of streamline trace],
            root_seeds [SeedGen])
        stream_opts: must have ds0 and max_length, maxit will be
            automatically calculated

    Returns:
        root points after following the fluid
    """
    times = np.array([grid.time for grid in vfile.iter_times(time_slice)])
    dt = np.roll(times, -1) - times  # Note: last element makes no sense

    grid_iter = vfile.iter_times(time_slice)
    return follow_fluid_generic(grid_iter, dt, initial_seeds, plot_function,
                                stream_opts, **kwargs)

def follow_fluid_generic(grid_iter, dt, initial_seeds, plot_function,
                         stream_opts, add_seed_cadence=0.0, add_seed_pts=None,
                         speed_scale=1.0):
    """Trace fluid elements

    Args:
        grid_iter (iterable): Some iterable that yields grids
        dt (float): Either one float for uniform dt, or a list
            where di[i] = grid_iter[i + 1].time - grid_iter[i].time
            The last element is repeated until grid_iter is exhausted
        initial_seeds: any SeedGen object
        plot_function: function that is called each time step,
            arguments should be exactly: (i [int], grid, v [Vector
            Field], v_lines [result of streamline trace],
            root_seeds [SeedGen])
        stream_opts: must have ds0 and max_length, maxit will be
            automatically calculated
        add_seed_cadence: how often to add the add_seeds points
        add_seed_pts: an n x 3 ndarray of n points to add every
            add_seed_cadence (xyz)
        speed_scale: speed_scale * v should be in units of ds0 / dt

    Returns:
        TYPE: Description
    """
    if not hasattr(dt, "__iter__"):
        dt = itertools.repeat(dt)
    else:
        dt = itertools.chain(dt, itertools.repeat(dt[-1]))

    root_seeds = initial_seeds

    # setup maximum number of iterations for a streamline
    if "ds0" in stream_opts and "max_length" in stream_opts:
        max_length = stream_opts["max_length"]
        ds0 = stream_opts["ds0"]
        stream_opts["maxit"] = (max_length // ds0) + 1
    else:
        raise KeyError("ds0 and max_length must be keys of stream_opts, "
                       "otherwise I don't know how to follow the fluid")

    # iterate through the time steps from time_slice
    for i, grid, dti in izip(itertools.count(), grid_iter, dt):
        if i == 0:
            last_add_time = grid.time

        root_pts = _follow_fluid_step(i, dti, grid, root_seeds,
                                      plot_function, stream_opts, speed_scale)

        # maybe add some new seed points to account for those that have left
        if (add_seed_pts is not None and
            abs(grid.time - last_add_time) >= add_seed_cadence):
            #
            root_pts = np.concatenate([root_pts, add_seed_pts.T], axis=1)
            last_add_time = grid.time
        root_seeds = seed.Point(root_pts)

    return root_seeds

def _follow_fluid_step(i, dt, grid, root_seeds, plot_function, stream_opts,
                       speed_scale):
    direction = int(dt / np.abs(dt))
    if direction >= 0:
        sl_direction = streamline.DIR_FORWARD
    else:
        sl_direction = streamline.DIR_BACKWARD

    logger.info("working on timestep {0} {1}".format(i, grid.time))
    v = grid["v"]
    logger.debug("finished reading V field")

    logger.debug("calculating new streamline positions")
    flow_lines = calc_streamlines(v, root_seeds,
                                  output=viscid.OUTPUT_STREAMLINES,
                                  stream_dir=sl_direction,
                                  **stream_opts)[0]

    logger.debug("done with that, now i'm plotting...")
    plot_function(i, grid, v, flow_lines, root_seeds)

    ############################################################
    # now recalculate the seed positions for the next timestep
    logger.debug("finding new seed positions...")
    root_pts = root_seeds.genr_points()
    valid_pt_inds = []
    for i in range(root_pts.shape[1]):
        valid_pt = True

        # get the index of the root point in teh 2d flow line array
        # dist = flow_lines[i] - root_pts[:, [i]]
        # root_ind = np.argmin(np.sum(dist**2, axis=0))
        # print("!!!", root_pts[:, i], "==", flow_lines[i][:, root_ind])
        # interpolate velocity onto teh flow line, and get speed too
        v_interp = cycalc.interp_trilin(v, seed.Point(flow_lines[i]))
        speed = np.sqrt(np.sum(v_interp * v_interp, axis=1))

        # this is a super slopy way to integrate velocity
        # keep marching along the flow line until we get to the next timestep
        t = 0.0
        ind = 0
        if direction < 0:
            flow_lines[i] = flow_lines[i][:, ::-1]
            speed = speed[::-1]

        while t < np.abs(dt):
            ind += 1
            if ind >= len(speed):
                # set valid_pt to True if you want to keep that point for
                # future time steps, but most likely if we're here, the seed
                # has gone out of our region of interest
                ind = len(speed) - 1
                valid_pt = False
                logger.info("OOPS: ran out of streamline, increase "
                            "max_length when tracing flow lines if this "
                            "is unexpected")
                break
            t += stream_opts["ds0"] / (speed_scale * speed[ind])

        root_pts[:, i] = flow_lines[i][:, ind]
        if valid_pt:
            valid_pt_inds.append(i)

    # remove seeds that have flown out of our region of interest
    # (aka, beyond the flow line we drew)
    root_pts = root_pts[:, valid_pt_inds]

    logger.debug("ok, done with all that :)")
    return root_pts

##
## EOF
##
