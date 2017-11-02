
from __future__ import print_function
from viscid_test_common import next_plot_fname

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import viscid
from viscid.readers import openggcm
from viscid.plot import vpyplot as vlt

# These two class definitions are examples of how to override a
# built-in reader and implement some convenience getters for
# derived quantities not in the file
class MyGGCMGrid(openggcm.GGCMGrid):
    mhd_to_gse_on_read = 'force'

    def _get_bcc(self):
        return self['b']

# So, you can make a class that derrives from an existing vFile type.
# That way you just define this class, and when it comes time to call
# load_file(file_name), you don't have to give it grid_type=...
# class MyGGCMFile(openggcm.GGCMFileXDMF):  # pylint: disable=W0223
#     # you can change the detector, or you can not and you will
#     # completely override the parent reader
#     _detector = r"^\s*.*\.(p[xyz]_[0-9]+|3d|3df|iof)" \
#                 r"(\.[0-9]{6})?\.(xmf|xdmf)\s*$"
#
#     # this is for injecting your convenience methods defined
#     # above
#     _grid_type = MyGGCMGrid


def run_test_3d(f, main__file__, show=False):
    vlt.clf()
    slc = "x=-20f:12f, y=0f"
    plot_kwargs = dict(title=True, earth=True)
    vlt.subplot(141)
    vlt.plot(f['pp'], slc, logscale=True, **plot_kwargs)
    vlt.subplot(142)
    vlt.plot(viscid.magnitude(f['bcc']), slc, logscale=True, **plot_kwargs)
    vlt.plot2d_quiver(f['v'][slc], step=5, color='y', pivot='mid', width=0.03,
                      scale=600)
    vlt.subplot(143)
    vlt.plot(f['jy'], slc, clim=(-0.005, 0.005), **plot_kwargs)
    vlt.streamplot(f['v'][slc], linewidth=0.3)
    vlt.subplot(144)
    vlt.plot(f['jy'], "x=7f:12f, y=0f, z=0f")

    plt.suptitle("3D File")
    vlt.auto_adjust_subplots(subplot_params=dict(top=0.9, wspace=1.3))
    plt.gcf().set_size_inches(10, 4)

    vlt.savefig(next_plot_fname(main__file__))
    if show:
        vlt.show()

def run_test_2d(f, main__file__, show=False):
    vlt.clf()
    slc = "x=-20f:12f, y=0f"
    plot_kwargs = dict(title=True, earth=True)
    vlt.subplot(141)
    vlt.plot(f['pp'], slc, logscale=True, **plot_kwargs)
    vlt.plot(np.abs(f['psi']), style='contour', logscale=True, levels=30,
             linewidths=0.8, colors='grey', linestyles='solid', colorbar=None,
             x=(-20, 12))
    vlt.subplot(142)
    vlt.plot(viscid.magnitude(f['bcc']), slc, logscale=True, **plot_kwargs)
    vlt.plot2d_quiver(f['v'][slc], step=5, color='y', pivot='mid', width=0.03,
                      scale=600)
    vlt.subplot(143)
    vlt.plot(f['jy'], slc, clim=[-0.005, 0.005], **plot_kwargs)
    vlt.streamplot(f['v'][slc], linewidth=0.3)
    vlt.subplot(144)
    vlt.plot(f['jy'], "x=7f:12f, y=0f, z=0f")

    plt.suptitle("2D File")
    vlt.auto_adjust_subplots(subplot_params=dict(top=0.9, wspace=1.3))
    plt.gcf().set_size_inches(10, 4)

    vlt.savefig(next_plot_fname(main__file__))
    if show:
        vlt.show()

def run_test_timeseries(f, main__file__, show=False):
    vlt.clf()

    ntimes = f.nr_times()
    t = [None] * ntimes
    pressure = np.zeros((ntimes,), dtype='f4')

    for i, grid in enumerate(f.iter_times()):
        t[i] = grid.time_as_datetime()
        pressure[i] = grid['pp']['x=10.0f, y=0.0f, z=0.0f']
    plt.plot(t, pressure)
    plt.ylabel('Pressure')

    dateFmt = mdates.DateFormatter('%H:%M:%S')
    plt.gca().xaxis.set_major_formatter(dateFmt)
    plt.gcf().autofmt_xdate()
    plt.gca().grid(True)

    plt.gcf().set_size_inches(8, 4)

    plt.savefig(next_plot_fname(main__file__))
    if show:
        vlt.mplshow()

def run_test_iof(f, main__file__, show=False):
    vlt.clf()

    fac_tot = 1e9 * f["fac_tot"]
    plot_args = dict(projection="polar",
                     lin=[-300, 300],
                     bounding_lat=35.0,
                     drawcoastlines=True,  # for basemap only
                     title="Total FAC\n",
                     gridec='gray',
                     label_lat=True,
                     label_mlt=True,
                     colorbar=dict(pad=0.1)  # pad the colorbar away from the plot
                    )

    ax1 = vlt.subplot(121, projection='polar')
    vlt.plot(fac_tot, ax=ax1, hemisphere='north', **plot_args)
    ax1.annotate('(a)', xy=(0, 0), textcoords="axes fraction",
                 xytext=(-0.1, 1.0), fontsize=18)

    ax2 = vlt.subplot(122, projection='polar')
    plot_args['gridec'] = False
    vlt.plot(fac_tot, ax=ax2, hemisphere="south", style="contourf",
             levels=50, extend="both", **plot_args)
    ax2.annotate('(b)', xy=(0, 0), textcoords="axes fraction",
                 xytext=(-0.1, 1.0), fontsize=18)

    vlt.auto_adjust_subplots(subplot_params=dict())
    plt.gcf().set_size_inches(8, 4)

    plt.savefig(next_plot_fname(main__file__))
    if show:
        vlt.mplshow()


##
## EOF
##
