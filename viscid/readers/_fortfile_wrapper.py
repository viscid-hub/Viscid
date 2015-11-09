# mostly stolen from pyggcm... thanks Matt

from viscid.readers import _fortfile

# FIXME: this my not play nicely in a multiprocessing environment
_available_units = list(range(10, 50))
_units_in_use = {}

class FortranFile(object):
    """
    small wrapper to allow the manipulation of fortran files from python
    """
    _unit = -1

    filename = None
    debug = None

    def __init__(self, name, debug=0):
        self.filename = name
        self.debug = debug

    # Make sure we close it when we're done
    def __del__(self):
        self.close()

    def open(self):
        self._unit = _available_units.pop()
        unit = _fortfile.fopen(self.filename, uu=self._unit, debug=self.debug)
        if unit == self._unit:
            _units_in_use[unit] = True
        else:
            raise RuntimeError("Fortran open file didn't return corretly")

    def close(self):
        if self.isopen:
            _fortfile.fclose(self._unit, debug=self.debug)
            del _units_in_use[self._unit]
            _available_units.append(self._unit)
            self._unit = -1

    def seek(self, offset, whence=0):
        assert self.isopen
        status = _fortfile.seek(self._unit, offset, whence)
        if status != 0:
            raise AssertionError("status != 0: {0}".format(status))
        return status

    def tell(self):
        assert self.isopen
        pos = _fortfile.tell(self._unit)
        assert pos >= 0
        return pos

    @property
    def isopen(self):
        if self._unit > 0:
            if bool(_fortfile.fisopen(self._unit)):
                return True
            else:
                raise RuntimeError("File has a valid unit, but fortran says "
                                   "it's closed?")
        return False

    @property
    def unit(self):
        return self._unit

    def rewind(self):
        _fortfile.frewind(self._unit, debug=self.debug)

    def advance_one_line(self):
        return _fortfile.fadvance_one_line(self._unit, debug=self.debug)

    def backspace(self):
        _fortfile.fbackspace(self._unit, debug=self.debug)

    def __enter__(self):
        if not self.isopen:
            self.open()
        return self

    def __exit__(self, exc_type, value, traceback):
        if self.isopen:
            self.close()
