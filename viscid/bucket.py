#!/usr/bin/env python

from __future__ import print_function

# from viscid.vutil import tree_prefix
try:
    from collections import OrderedDict
except ImportError:
    from viscid.compat.ordered_dict_backport import OrderedDict

from viscid import logger

class Bucket(object):
    """ This is an interface where  """
    _ordered = False

    _items = None  # keys are items, values are list of handles
    _handles = None  # keys are handles, values are items

    # if index handle, set_item adds this number as a handle and increments it
    # this is useful for hiding loads that are not user initiated, such as
    # an xdmf file loading an h5 file under the covers
    _int_counter = None

    def __init__(self, ordered=False):
        self._ordered = ordered

        if self._ordered:
            self._items = OrderedDict()
            self._handles = OrderedDict()
        else:
            self._items = {}
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

        if item not in self._items:
            if index_handle:
                handles += [self._int_counter]
                self._int_counter += 1
            self._items[item] = handles

            if len(handles) == 0:
                raise ValueError("item {0} must have at least one "
                                 "handle".format(item))

        for h in handles:
            # check if we're stealing a handle from another item
            if (h in self._handles) and  (item is not self._handles[h]):
                logger.warn("The handle {0} is being hijacked! Memory leak "
                             "could ensue.".format(h))
                # romove handle from old item, since this check is here,
                # there sholdn't be 2 items with the same handle in the
                # items dict
                old_item = self._handles[h]
                self._items[old_item].remove(h)
                if len(self._items[h]) == 0:
                    self.remove_item(old_item)

            self._handles[h] = item

        return None

    def remove_item(self, item):
        """ remove item, raises ValueError if item is not found """
        handles = self._items[item]
        del self._items[item]
        for h in handles:
            del self._handles[h]

    def remove_item_by_handle(self, handle):
        """ remove item by handle, raises KeyError if handle is not found """
        self.remove_item(self._handles[handle])

    def remove_all_items(self):
        """ unload all items """
        # TODO: maybe unload things explicitly?
        if self._ordered:
            self._items = OrderedDict()
            self._handles = OrderedDict()
        else:
            self._items = {}
            self._handles = {}

    def items_as_list(self):
        return list(self._items.keys())

    def get_primary_handles(self):
        """Return a list of the first handles for all items"""
        return [handles[0] for handles in self._items.values()]

    def handle_string(self, prefix=""):
        """ return string representation of handles and items """
        # this is inefficient, but probably doesn't matter
        s = ""
        for item, handles in self._items.items():
            hands = [repr(h) for h in handles]
            s += "{0}handles: {1}\n".format(prefix, ", ".join(hands))
            s += "{0}  item: {1}\n".format(prefix, str(item))
        return s

    def print_tree(self, prefix=""):
        print(self.handle_string(prefix=prefix), end='')

    def __getitem__(self, handle):
        return self._handles[handle]

    def __setitem__(self, key, value):
        if isinstance(key, (list, tuple)):
            key = list(key)
        elif key is not None:
            key = [key]
        self.set_item(key, value)

    def __delitem__(self, handle):
        try:
            self.remove_item_by_handle(handle)
        except KeyError:
            # maybe we are asking to remove an item explicitly
            self.remove_item(handle)

    def __iter__(self):
        return self._items.keys().__iter__()

    def __contains__(self, handle):
        return handle in self._handles or handle in self._items

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
