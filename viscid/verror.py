#!/usr/bin/env python

from __future__ import print_function


class UnimportedModule(object):
    def __init__(self, exception, msg="", **attrs):
        attrs["exception"] = exception
        attrs["msg"] = msg

        def _fall_over():
            if self.msg:
                from viscid import logger
                logger.critical(self.msg)
            raise self.exception
        attrs['_fall_over'] = _fall_over

        for attrname, value in attrs.items():
            super(UnimportedModule, self).__setattr__(attrname, value)

    def __getattr__(self, name):
        try:
            return super(UnimportedModule, self).__getattr__(name)
        except AttributeError:
            self._fall_over()
    def __setattr__(self, name, value):
        self._fall_over()


class DeferredImportError(ImportError):
    pass


class BackendNotFound(RuntimeError):
    pass


class KeyboardInterruptError(Exception):
    pass

##
## EOF
##
