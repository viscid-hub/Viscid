#!/usr/bin/env python

from __future__ import print_function

class FileNotFound(RuntimeError):
    pass

class FieldNotFound(RuntimeError):
    pass

class BackendNotFound(RuntimeError):
    pass

class KeyboardInterruptError(Exception):
    pass

##
## EOF
##
