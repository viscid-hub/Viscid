#!/usr/bin/env python

from __future__ import print_function
from .vutil import tree_prefix
import logging

class Bucket(object):
    """ This is an interface where  """
    _items = None
    _handles = None
    # if index handle, set_item adds this number as a handle and increments it
    # this is useful for hiding loads that are not user initiated, such as
    # an xdmf file loading an h5 file under the covers
    _int_counter = None

    def __init__(self):
        self._items = []
        self._handles = {}
        self._int_counter = 0

    def set_item(self, handles, item, index_handle=True):
        """ if index_handle is true then the index of item will be included as
            a handle making the bucket indexable like a list """
        # found = False
        if handles is None:
            handles = []
        if not isinstance(handles, list):
            raise TypeError("handle must by of list type")

        try:
            self.index(item)
        except ValueError:
            if index_handle:
                handles += [self._int_counter] # handles += [onum]
                self._int_counter += 1
            self._items.append(item)

        self._set_handles(handles, item)
        return None

    def remove_item(self, item):
        """ remove item, raises ValueError if item is not found """
        self.remove_item_onum(self.index(item))
    def remove_item_handle(self, handle):
        """ remove item by handle, raises KeyError if handle is not found """
        self.remove_item_onum(self._handles[handle])
    def remove_item_onum(self, onum):
        for k in self._handles.keys():
            if self._handles[k] == onum:
                self._handles.pop(k)
        self._items[onum] = None

    def remove_all_items(self):
        """ unload all items """
        for i in range(len(self._items)):
            self.remove_item_onum(i)

    def get_item(self, handle):
        """ look up the file using the handle dictionary, raises KeyError
            if handle not found """
        return self._items[self._handles[handle]]

    def items_as_list(self):
        return self._items

    def index(self, item):
        """ find index of item, comparison is made using object id """
        for i, it in enumerate(self._items):
            if item is it:
                return i
        raise ValueError("item's object id not in list")

    def handle_string(self, prefix=""):
        """ return string representation of handles and items """
        # this is inefficient, but probably doesn't matter
        s = ""
        for i, item in enumerate(self._items):
            hands = [repr(h) for h, onum in self._handles.items() if onum == i]
            s += "{0}handles: {1}\n".format(prefix, ", ".join(hands))
            s += "{0}  item: {1}\n".format(prefix, str(item))
        return s

    def print_tree(self, prefix=""):
        print(self.handle_string(prefix=prefix), end='')

    def _set_handles(self, handles, item):
        """ this should only be called for items that are in _items...
            a ValueError will result otherwise """
        onum = self.index(item)
        for h in handles:
            if (h in self._handles) and (not item is self._items[onum]):
                logging.warn("The handle {0} is being hijacked!".format(h))
            self._handles[h] = onum
        return None

    def __getitem__(self, handle):
        return self.get_item(handle)

    def __setitem__(self, key, value):
        if isinstance(key, (list, tuple)):
            key = list(key)
        elif key is not None:
            key = [key]
        self.set_item(key, value)

    def __delitem__(self, handle):
        self.remove_item_handle(handle)

    def __iter__(self):
        return self._items.__iter__()

    def __contains__(self, handle):
        return handle in self._handles
    def __len__(self):
        return len(self._items)
    def __str__(self):
        return self.handle_string()

if __name__ == '__main__':
    import os
    import code
    import viscid

    _viscid_root = os.path.dirname(viscid.__file__)
    sample_dir = _viscid_root + "/../../sample/"
    sample_prefix = sample_dir + "local_0001"
    fnpy0 = sample_prefix + ".py_0.xdmf"
    fn3dfa = sample_prefix + ".3df.xdmf"
    fn3df = sample_prefix + ".3df.004200.xdmf"
    fniof = sample_prefix + ".iof.004200.xdmf"
    fnasc = sample_dir + "test.asc"
    fnascts = sample_dir + "test_time.asc"

    # fm = get_file_bucket()

    # print("load {0}".format(fnasc))
    # fasc = fm.add(fnasc)
    # print("load {0}".format(fnascts))
    # fasc_time = fm.add(fnascts)

    # print("load 3df.xdmf")
    # d3da = fm.load(fn3dfa)
    # print("load 3df.004200.xdmf")
    # f3d = load_file(fn3df)
    # print("load py_0.xdmf")
    # fpy0 = load_file(fnpy0)
    # print("load test.xdmf")
    # ftest = fm.load('../sample/test.xdmf')

    code.interact(local=locals())

    print("done")

# a=re.findall('(\S+)\s*=\s*(?:[\'"](.*?)[\'"]|(.*?))(?:\s|$)',
#    'key="value" delim=" " socrates k=v', flags=0); print(a)

##
## EOF
##
