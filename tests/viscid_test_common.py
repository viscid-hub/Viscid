"""
Usage:

from viscid_test_common import test_dir, sample_dir, plot_dir, get_test_name
test_name = get_test_name(__file__)
"""

from __future__ import print_function
import os
import sys


CODE_XFAIL = 0xf0

# handle making unique plot filenames
NPLOT = {}

# useful paths
test_dir = os.path.dirname(__file__)
sample_dir = test_dir + "/../sample"
plot_dir = test_dir + "/plots"
ref_dir = test_dir + "/ref_plots"

# add Viscid root to python path if it's not already there
viscid_root = os.path.realpath(test_dir + '/../')

try:
    if int(os.environ["VISCID_TEST_INPLACE"]):
        sys.path.insert(0, viscid_root)
    else:
        # VISCID_TEST_INPLACE=0 means use the first viscid in PYTHONPATH
        pass
except KeyError:
    # if VISCID_TEST_INPLACE is not specified, use the first viscid in
    # PYTHONPATH, but add viscid_root as a fallback at the end
    sys.path.append(viscid_root)

# set default plot style
import viscid  # pylint: disable=unused-import
try:
    from matplotlib import style
    style.use("seaborn-notebook")
except (ImportError, ValueError):
    from viscid.plot import vseaborn
    vseaborn.context = "notebook"

# do some common string handling
def get_test_name(main__file__):
    """extract the name of the main test script

    Args:
        main__file__ (str): __file__ of the main script

    Returns:
        str: Name of the test
    """
    test_fname = os.path.splitext(os.path.basename(main__file__))[0]
    test_name = test_fname

    if test_name.startswith("test_"):
        test_name = test_name[len("test_"):]
    if test_name.endswith(".py"):
        test_name = test_name[:-len(".py")]

    return test_name

def next_plot_fname(main__file__, series='', fmt='png'):
    """Make a new unique plot filename for a given test

    Args:
        main__file__ (str): __file__ of the main script

    Returns:
        str: Unique file name for a given figure of a given test
    """
    if series not in NPLOT:
        NPLOT[series] = 0

    test_name = get_test_name(main__file__)
    if series:
        sseries = series + "_"
    else:
        sseries = series
    name = "{0}/{1}-{2}{3:03d}.{4}".format(plot_dir, test_name, sseries,
                                           NPLOT[series], fmt)
    NPLOT[series] += 1
    return name

##
## EOF
##
