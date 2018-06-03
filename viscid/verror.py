#!/usr/bin/env python

from __future__ import print_function


__all__ = ['UnimportedModule', 'DeferredImportError', 'BackendNotFound',
           'NoBasetimeError', 'KeyboardInterruptError']


class UnimportedModule(object):
    """Proxy object for unimported modules"""
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
    def __call__(self):
        self._fall_over()


class DeferredImportError(ImportError):
    """So lack of an optional dependency doesn't make viscid unimportable"""
    pass


class BackendNotFound(RuntimeError):
    """Calculator backend not installed"""
    pass


class NoBasetimeError(Exception):
    """When a dataset is trying to get time as datetime but has no basetime"""
    def __init__(self, msg):
        super(NoBasetimeError, self).__init__(msg)


class KeyboardInterruptError(Exception):
    """Deprecated and unused"""
    pass

##
## EOF
##
