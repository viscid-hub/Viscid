#!/usr/bin/env python
"""Call out to external tools like ffmpeg and meshlab"""

from __future__ import print_function
import os
import subprocess as sub


__all__ = ['make_animation', 'meshlab_convert']


def make_animation(movie_fname, prefix, framerate=5, qscale=2, keep=False,
                   args=None, frame_idx_fmt="_%06d", program="ffmpeg",
                   yes=False):
    """ make animation by calling program (only ffmpeg works for now) using
    args, which is a namespace filled by the argparse options from
    add_animate_arguments. Plots are expected to be named
    ${args.prefix}_000001.png where the number is in order from 1 up """
    if args is not None:
        prefix = args.prefix
        framerate = args.framerate
        qscale = args.qscale
        movie_fname = args.animate
        keep = args.keep

    if movie_fname:
        cmd = "yes | {0}".format(program) if yes else program
        if program == "ffmpeg":
            sub.Popen("{0} -r {1} -i {3}{4}.png -pix_fmt yuv420p "
                      "-qscale {2} {5}".format(cmd, framerate, qscale, prefix,
                                               frame_idx_fmt, movie_fname),
                      shell=True).communicate()
    if movie_fname is None and prefix is not None:
        keep = True
    if not keep:
        sub.Popen("rm -f {0}_*.png".format(prefix), shell=True).communicate()
    return None

def meshlab_convert(fname, fmt="dae", quiet=True):
    """Run meshlabserver to convert 3D mesh files

    Uses `MeshLab <http://meshlab.sourceforge.net/>`_, which is a great
    little program for playing with 3D meshes. The best part is that
    OS X's Preview can open the COLLADA (`*.dae`) format. How cool is
    that?

    Args:
        fname (str): file to convert
        fmt (str): extension of result, defaults to COLLADA format
        quiet (bool): redirect output to :py:attr:`os.devnull`

    Returns:
        None
    """
    iname = fname
    oname = '.'.join(iname.split('.')[:-1]) + "." + fmt.strip()
    redirect = "&> {0}".format(os.devnull) if quiet else ""
    cmd = ("meshlabserver -i {0} -o {1} -m vc vn fc fn {2}"
           "".format(iname, oname, redirect))
    sub.Popen(cmd, shell=True, stdout=None, stderr=None)

##
## EOF
##
