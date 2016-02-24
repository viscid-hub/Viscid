#!/usr/bin/env python

from __future__ import print_function

class DeferredImportError(ImportError):
    pass

class BackendNotFound(RuntimeError):
    pass

class KeyboardInterruptError(Exception):
    pass

##
## EOF
##
