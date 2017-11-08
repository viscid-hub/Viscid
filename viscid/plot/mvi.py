
import viscid

viscid.logger.warning("The module viscid.plot.mvi has moved to "
                      "viscid.plot.vlab since mvi is a common shortcut for "
                      "mayavi itself. Your import should rather look like: "
                      "`from viscid.plot import vlab` Thanks!")

from viscid.plot.vlab import *  # pylint: disable=wildcard-import,unused-wildcard-import,wrong-import-position

##
## EOF
##
