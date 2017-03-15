#!/usr/bin/env python

from __future__ import print_function
import sys

# from viscid.vutil import tree_prefix
from viscid.compat import OrderedDict
from viscid import logger

class Bucket(object):
    """ This is basically a glorified dict

    It's a convenient dict-like object if you want lots of
    keys for a given value.

    NOTE:
        You can add non-hashable items, but this is poorly tested.
        When adding / removing non-hashable items (items, not handles)
        the comparison is done using the object's id. This is
        fundamentally different than using an object's __hash__, but
        it should be fairly transparent.
    """
    _ordered = False

    _ref_count = None  # keys are hashable items, values are # of times item was added
    _hash_lookup = None  # keys are hashable items, values are actual items
    _handles = None  # keys are hashable items, values are list of handles
    _items = None  # keys are handles, values are actual items

    # if index handle, set_item adds this number as a handle and increments it
    # this is useful for hiding loads that are not user initiated, such as
    # an xdmf file loading an h5 file under the covers
    _int_counter = None

    def __init__(self, ordered=False):
        self._ordered = ordered

        self._set_empty_dicts()
        self._int_counter = 0

    def _set_empty_dicts(self):
        if self._ordered:
            self._ref_count = OrderedDict()
            self._hash_lookup = OrderedDict()
            self._handles = OrderedDict()
            self._items = OrderedDict()
        else:
            self._ref_count = {}
            self._hash_lookup = {}
            self._handles = {}
            self._items = {}

    @staticmethod
    def _make_hashable(item):
        try:
            hash(item)
            return item
        except TypeError:
            return "<{0} @ {1}>".format(type(item), hex(id(item)))

    def items(self):
        for hashable_item, item in self._hash_lookup.items():
            yield self._handles[hashable_item], item

    def keys(self):
        return self._handles.values()

    def values(self):
        return self._hash_lookup.values()

    def set_item(self, handles, item, index_handle=True, _add_ref=False):
        """ if index_handle is true then the index of item will be included as
            a handle making the bucket indexable like a list """
        # found = False
        if handles is None:
            handles = []
        if not isinstance(handles, list):
            raise TypeError("handle must by of list type")

        # make sure we have a hashable "item" for doing reverse
        # lookups of handles using an item
        hashable_item = self._make_hashable(item)
        if hashable_item not in self._hash_lookup:
            if index_handle:
                handles += [self._int_counter]
                self._int_counter += 1

        handles_added = []
        for h in handles:
            # check if we're stealing a handle from another item
            try:
                hash(h)
            except TypeError:
                logger.error("A bucket says handle '{0}' is not hashable, "
                             "ignoring it".format(h))
                continue

            if (h in self._items) and (item is self._items[h]):
                continue
            elif h in self._items:
                logger.error("The handle '{0}' is being hijacked! Memory leak "
                             "could ensue.".format(h))
                # romove handle from old item, since this check is here,
                # there sholdn't be 2 items with the same handle in the
                # items dict
                old_item = self._items[h]
                old_hashable_item = self._make_hashable(old_item)
                self._handles[old_hashable_item].remove(h)
                if len(self._handles[old_hashable_item]) == 0:
                    self.remove_item(old_item)
            self._items[h] = item
            handles_added.append(h)

        try:
            self._handles[hashable_item] += handles_added
            if _add_ref:
                self._ref_count[hashable_item] += 1
        except KeyError:
            if len(handles_added) == 0:
                logger.error("No valid handles given, item '{0}' not added to "
                             "bucket".format(hashable_item))

            else:
                self._handles[hashable_item] = handles_added
                self._hash_lookup[hashable_item] = item
                self._ref_count[hashable_item] = 1

        return None

    def _remove_item(self, item):
        """remove item no matter what

        You may want to use remove_
        , raises ValueError if item is not found """
        hashable_item = self._make_hashable(item)
        handles = self._handles[hashable_item]
        for h in handles:
            del self._items[h]
        del self._hash_lookup[hashable_item]
        del self._handles[hashable_item]
        del self._ref_count[hashable_item]

    def _remove_item_by_handle(self, handle):
        self._remove_item(self._items[handle])

    def remove_item(self, item):
        self._remove_item(item)

    def remove_item_by_handle(self, handle):
        """ remove item by handle, raises KeyError if handle is not found """
        self.remove_item(self._items[handle])

    def remove_reference(self, item, _ref_count=1):
        hashable_item = self._make_hashable(item)
        try:
            self._ref_count[hashable_item] -= _ref_count
        except KeyError:
            item = self[item]
            hashable_item = self._make_hashable(item)
            if _ref_count:
                self._ref_count[hashable_item] -= _ref_count
            else:
                self._ref_count[hashable_item] = 0

        # FIXME: unload_all_files breaks this assert check... probably a bug
        # assert self._ref_count[hashable_item] >= 0, \
        #     "problem with bucket ref counting {0}".format(hashable_item)
        if self._ref_count[hashable_item] <= 0:
            self._remove_item(item)

    def remove_all_items(self):
        """ unload all items """
        self._set_empty_dicts()

    def items_as_list(self):
        return list(self._hash_lookup.values())

    def get_primary_handles(self):
        """Return a list of the first handles for all items"""
        return [handles[0] for handles in self._handles.values()]

    def handle_string(self, prefix=""):
        """ return string representation of handles and items """
        # this is inefficient, but probably doesn't matter
        s = ""
        for item, handles in self._handles.items():
            hands = [repr(h) for h in handles]
            s += "{0}handles: {1}\n".format(prefix, ", ".join(hands))
            s += "{0}  item: {1}\n".format(prefix, str(item))
        return s

    def print_tree(self, prefix=""):
        print(self.handle_string(prefix=prefix), end='')

    def __getitem__(self, handle):
        return self._items[handle]

    def __setitem__(self, key, value):
        if isinstance(key, (list, tuple)):
            key = list(key)
        elif key is not None:
            key = [key]
        self.set_item(key, value)

    def __delitem__(self, handle):
        try:
            self.remove_item_by_handle(handle)
        except (KeyError, TypeError):
            # maybe we are asking to remove an item explicitly
            self.remove_item(handle)

    def __iter__(self):
        return self.values().__iter__()

    def contains_item(self, item):
        hashable_item = self._make_hashable(item)
        return hashable_item in self._handles

    def contains_handle(self, handle):
        try:
            return handle in self._items
        except TypeError:
            return False

    def __contains__(self, handle):
        return self.contains_handle(handle) or self.contains_item(handle)

    def __len__(self):
        return len(self._hash_lookup)

    def __str__(self):
        return self.handle_string()


def _main():
    import os
    import viscid

    sample_dir = os.path.join(viscid.sample_dir, "local_0001")
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

    viscid.interact()

    print("done")

    # a=re.findall('(\S+)\s*=\s*(?:[\'"](.*?)[\'"]|(.*?))(?:\s|$)',
    #    'key="value" delim=" " socrates k=v', flags=0); print(a)

    return 0

if __name__ == '__main__':
    sys.exit(_main())

##
## EOF
##
