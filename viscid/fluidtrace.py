#!/usr/bin/env python
"""Trace fluid elements in time"""
# pylint: disable=deprecated-method

from __future__ import print_function
import itertools
import sys

import numpy as np

import viscid
from viscid.compat import izip, OrderedDict
from viscid.calculator import calc_streamlines
from viscid.calculator import cycalc
from viscid.calculator import streamline
from viscid import logger
from viscid import seed


__all__ = ['SeedCurator', 'ContinuousCurator', 'ReplacementCurator',
           'follow_fluid']


class SeedCurator(object):
    """Base class for adding/removing seeds from fluid following

    This class just removes seeds that go out of bounds, but provides
    the _update_delmask_oob method which is probably useful to all
    """

    def __init__(self, obound0=None, obound1=None, ibound=None, obound_r=None):
        self.obound_xl = None
        self.obound_xh = None
        self.obound0 = obound0
        self.obound1 = obound1
        self.ibound = ibound
        self.obound_r = obound_r

    def _prepare_xl_xh(self, v_field):
        # Jump through some hoops so that obound0 and obound1 are expanded
        # to 3d in the same way as grid.crds
        if self.obound_xl is None:
            assert self.obound_xh is None
            if self.obound0 is None:
                self.obound0 = v_field.crds.xl_nc
            if self.obound1 is None:
                self.obound1 = v_field.crds.xh_nc

            _invar_dim_mask = np.array(v_field.sshape) == 1
            self.obound0[_invar_dim_mask] = -np.inf
            self.obound1[_invar_dim_mask] = np.inf

            if (np.any(np.isinf(self.obound0)) or np.any(np.isinf(self.obound1))
                or np.any(np.isnan(self.obound0)) or np.any(np.isnan(self.obound1))):
                np_err_state = {'invalid': 'ignore'}
            else:
                np_err_state = {}

            with np.errstate(**np_err_state):
                _axes = v_field.crds.axes
                crds = viscid.arrays2crds(np.vstack([self.obound0, self.obound1]).T,
                                          crd_names=_axes).atleast_3d(cc=True)
                self.obound_xl = crds.xl_nc
                self.obound_xh = crds.xh_nc
        else:
            assert self.obound_xh is not None

    def _update_delmask_oob(self, v_field, seeds, delmask=None):
        """delmask |= where seeds are out of bounds"""
        self._prepare_xl_xh(v_field)

        if delmask is None:
            delmask = np.zeros([seeds.shape[1]], dtype='bool')

        delmask |= np.any(np.bitwise_or(seeds.T < self.obound_xl,
                                        seeds.T > self.obound_xh), axis=1)

        if self.ibound or self.obound_r:
            rsq = np.sum(seeds**2, axis=0)
            if self.ibound:
                delmask |= rsq < self.ibound**2
            if self.obound_r:
                delmask |= rsq > self.obound_r**2

        return delmask

    def update(self, v_field, seeds, delmask=None, time=None):
        """add/remove seeds"""
        delmask = self._update_delmask_oob(v_field, seeds, delmask=delmask)
        return np.array(seeds[:, ~delmask])


class ContinuousCurator(SeedCurator):
    """Add seeds to the same places at a steady cadence"""
    def __init__(self, new_seeds, cadence=0.1, **kwargs):
        self.new_seeds = np.array(new_seeds)
        self.cadence = cadence
        self.last_run = None
        super(ContinuousCurator, self).__init__(**kwargs)

    def update(self, v_field, seeds, delmask=None, time=None):
        """remove out-of-bounds seeds, then add the seeds at cadence"""
        time = v_field.time if time is None else time
        if self.last_run is None:
            self.last_run = time

        if time - self.last_run > self.cadence:
            delmask = self._update_delmask_oob(v_field, seeds, delmask=delmask)
            seeds = np.concatenate([seeds[:, ~delmask], self.new_seeds], axis=1)
            self.last_run = time
        return seeds


class ReplacementCurator(SeedCurator):
    """Remove out-of-bounds seeds, then reset them"""
    def __init__(self, orig_seeds=None, **kwargs):
        self.orig_seeds = np.array(orig_seeds) if orig_seeds is not None else None
        super(ReplacementCurator, self).__init__(**kwargs)

    def update(self, v_field, seeds, delmask=None, time=None):
        """Remove out-of-bounds seeds, then reset them"""
        if self.orig_seeds is None:
            self.orig_seeds = np.array(seeds)
        delmask = self._update_delmask_oob(v_field, seeds, delmask=delmask)
        # for any seeds marked for deletion, replace them with their
        # original selves
        seeds[:, delmask] = self.orig_seeds[:, delmask]
        return seeds


def fluid_callback_template(i, t, seeds=None, v_field=None, anc_fields=None,
                            grid0=None, grid1=None, streamlines=None,
                            series_fname=None, show=False, **kwargs):
    pass

def default_fluid_callback(i, t, seeds=None, v_field=None, anc_fields=None,
                           grid0=None, grid1=None, streamlines=None,
                           series_fname=None, show=False, **kwargs):
    from viscid.plot import vpyplot as vlt
    from matplotlib import pyplot as plt

    _, ax0 = plt.subplots(1, 1)
    vlt.plot(viscid.magnitude(v_field).atmost_nd(2), ax=ax0)
    vlt.scatter_2d(seeds, ax=ax0)
    vlt.streamplot(v_field.atmost_nd(2))
    vlt.plot_lines2d(streamlines, ax=ax0)
    plt.title('{0:8.3f}'.format(t))
    if series_fname:
        plt.savefig('{0}_{1:06d}.png'.format(series_fname, i))
    if show:
        vlt.show()
    else:
        plt.close()

def follow_fluid(dset, initial_seeds, time_slice=slice(None),
                 curator=None, callback=default_fluid_callback,
                 speed_scale=1.0, dt=None, tstart=None, tstop=None,
                 duration=None, dt_interp=None,
                 v_key='v', anc_keys=(), fld_slc=Ellipsis,
                 stream_opts={}, callback_kwargs={}):
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
    curator = SeedCurator() if curator is None else curator

    grids = [grid for grid in dset.iter_times(time_slice)]
    times = [g.time for g in grids]

    slc_range = dset.tslc_range(time_slice)
    time_slice_dir = np.sign(times[-1] - times[0]).astype('f')
    slice_min_dt = 1.0 if len(times) <= 1 else np.min(np.abs(np.diff(times)))

    # figure out direction (forward / backward)
    if tstart is not None and tstop is not None:
        tdir = np.sign(tstop - tstart).astype('f')
    elif (dt is not None and dt < 0) or (duration is not None and duration < 0):
        tdir = -1.0
    else:
        tdir = 1.0 if time_slice_dir == 0.0 else time_slice_dir

    # enforce that grids and times arrays are reordered to match tdir
    if (tdir > 0 and time_slice_dir < 0) or (tdir < 0 and time_slice_dir > 0):
        grids = grids[::-1]
        times = times[::-1]
        slc_range = slc_range[::-1]
        time_slice_dir *= -1

    # set tstart and tstop if they're not already given
    if tstart is None:
        tstart = slc_range[0]

    if tstop is None:
        if duration is not None:
            tstop = tstart + tdir * np.abs(duration)
        else:
            tstop = slc_range[1]

    # set dt if they're not given
    dt = np.abs(dt) if dt is not None else slice_min_dt
    dt_interp = np.abs(dt_interp) if dt_interp is not None else dt

    # ------ main loop
    fld_keys = [v_key] + list(anc_keys)

    times = np.array(times)
    t = tstart
    if np.any(np.sign(np.diff(times)) != tdir):
        raise RuntimeError("times is not monotonic")

    i = 0
    seeds = initial_seeds.get_points()

    while tdir * (t - tstop) <= 0.0:
        idx0 = max(np.sum(tdir * (times - t) < 0.0) - 1, 0)
        idx1 = min(idx0 + 1, len(grids) - 1)
        time0, grid0 = times[idx0], grids[idx0]
        time1, grid1 = times[idx1], grids[idx1]

        frac_interp = 0.0 if time0 == time1 else (t - time0) / (time1 - time0)

        # get / calculate fields for each key at the current time
        if grid0 is grid1:
            flds = [grid0[key] for key in fld_keys]
        else:
            a = frac_interp
            b = 1.0 - frac_interp
            flds = [viscid.axpby(a, grid0[k][fld_slc], b, grid1[k][fld_slc])
                    for k in fld_keys]
        anc_fields = OrderedDict([(k, v) for k, v in zip(anc_keys, flds[1:])])

        t_next_interp = t + tdir * dt_interp

        while tdir * (t - t_next_interp) < 0 and tdir * (t - tstop) <= 0.0:
            if 'method' not in stream_opts:
                stream_opts['method'] = 'rk45'
            vpaths = viscid.calc_streamlines(tdir * speed_scale * flds[0], seeds,
                                             max_t=dt,
                                             output=viscid.OUTPUT_STREAMLINES,
                                             stream_dir=viscid.DIR_FORWARD,
                                             **stream_opts)[0]

            callback(i, t, seeds=seeds, v_field=flds[0], anc_fields=anc_fields,
                     grid0=grid0, grid1=grid1, streamlines=vpaths,
                     **callback_kwargs)
            i += 1

            # prepare seeds for next iteration
            for iseed in range(seeds.shape[1]):
                seeds[:, iseed] = vpaths[iseed][:, -1]
            seeds = curator.update(flds[0], seeds, time=t)
            t += tdir * dt

def _main():
    import os
    from viscid.plot import vpyplot as vlt

    grid = viscid.grid.Grid(time=0.0)
    crds = viscid.arrays2crds([np.linspace(-1, 1, 32), np.linspace(-1, 1, 32)])
    grid.add_field(viscid.full(crds, np.nan, name='V'))

    seeds0 = viscid.Circle(p0=[0.0, 0.0, 0.0], r=0.8, n=25).get_points()
    seeds1 = viscid.Circle(p0=[0.0, 0.0, 0.0], r=1.1, n=25).get_points()
    seeds3 = viscid.Line([-0.5, 0, 0], [0.5, 0, 0], n=5)
    delmask = np.zeros([seeds0.shape[1]], dtype='bool')

    curator = viscid.SeedCurator()
    seeds2 = curator.update(grid['V'], np.array(seeds1), delmask=delmask)
    vlt.plt.scatter(seeds1[0], seeds1[1], c=[0.0, 1.0, 0.0])
    vlt.plt.scatter(seeds2[0], seeds2[1])
    vlt.plt.axhline(-1); vlt.plt.axvline(-1)  # pylint: disable=multiple-statements
    vlt.plt.axhline(1); vlt.plt.axvline(1)  # pylint: disable=multiple-statements
    vlt.show()

    curator = viscid.ReplacementCurator(seeds0)
    seeds2 = curator.update(grid['V'], np.array(seeds1), delmask=delmask)
    vlt.plt.scatter(seeds1[0], seeds1[1], c=[0.0, 1.0, 0.0])
    vlt.plt.scatter(seeds2[0], seeds2[1])
    vlt.plt.axhline(-1); vlt.plt.axvline(-1)  # pylint: disable=multiple-statements
    vlt.plt.axhline(1); vlt.plt.axvline(1)  # pylint: disable=multiple-statements
    vlt.show()

    curator = viscid.ContinuousCurator(seeds3, cadence=-1)
    seeds2 = curator.update(grid['V'], np.array(seeds1), delmask=delmask)
    vlt.plt.scatter(seeds1[0], seeds1[1], c=[0.0, 1.0, 0.0])
    vlt.plt.scatter(seeds2[0], seeds2[1])
    vlt.plt.axhline(-1); vlt.plt.axvline(-1)  # pylint: disable=multiple-statements
    vlt.plt.axhline(1); vlt.plt.axvline(1)  # pylint: disable=multiple-statements
    vlt.show()

    target_dir = os.path.join(os.path.expanduser('~'), 'Desktop', 'fluid_movie')
    print("Attempting to make a movie in:", target_dir)
    f = viscid.load_file("~/dev/stage/otico_001/otico*.3d.xdmf")

    xl, xh = f.get_grid().xl_nc, f.get_grid().xh_nc
    seeds0 = viscid.Circle(p0=0.5 * (xl + xh), r=0.2 * np.max(xh - xl),
                           pole=(0, 0, 1), n=10)
    seeds1 = viscid.Circle(p0=0.5 * (xl + xh), r=0.4 * np.max(xh - xl),
                           pole=(0, 0, 1), n=10)
    seeds = viscid.Point(np.concatenate([seeds0.get_points(),
                                         seeds1.get_points()], axis=1))

    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)
    target_fname = os.path.join(target_dir, 'fluid')
    viscid.follow_fluid(f, seeds, dt=0.0101926 / 2,
                        callback_kwargs=dict(show=False,
                                             series_fname=target_fname))

    return 0

if __name__ == "__main__":
    sys.exit(_main())

##
## EOF
##
