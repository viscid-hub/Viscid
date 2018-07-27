#!/usr/bin/env python
"""Test viscid.rotation module"""

from __future__ import print_function
import sys

import viscid


def _main():
    return viscid.rotation._check_all(quick=True)

if __name__ == "__main__":
    sys.exit(_main())

##
## EOF
##
