
import viscid

viscid.logger.warning("The module viscid.plot.mpl has moved to "
                      "viscid.plot.vpyplot since mpl is a common shortcut for "
                      "matplotlib itself. Your import should rather look like: "
                      "`from viscid.plot import vpyplot as vlt` Thanks!")

from viscid.plot.vpyplot import *  # pylint: disable=wildcard-import,unused-wildcard-import,wrong-import-position

##
## EOF
##
