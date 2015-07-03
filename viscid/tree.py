"""Classes for elements along the tree from files to fields"""

from __future__ import print_function
from string import ascii_letters
from datetime import datetime, timedelta

from viscid.compat import string_types
from viscid.vutil import format_time as generic_format_time

class Node(object):
    """Base class for Datasets and Grids"""

    name = None
    time = None
    _info = None
    parents = None

    def __init__(self, name=None, time=None, info=None, parents=None):
        if name is None:
            name = "<{0} @ {1}>".format(self.__class__.__name__, hex(id(self)))
        self.name = name
        self.time = time

        if info is None:
            info = dict()
        self._info = info

        if parents is None:
            parents = []
        if not isinstance(parents, (list, tuple)):
            parents = [parents]
        self.parents = parents

    def prepare_child(self, obj):
        if not self in obj.parents:
            obj.parents.append(self)

    def tear_down_child(self, obj):
        try:
            obj.parents.remove(self)
        except ValueError:
            pass

    def _parent_bfs(self, condition, getvalue=None):
        """Breadth first search of parent

        Args:
            condition (callable): function for match condition, must
                take 2 arguments, the first being the object in
                question, and the 2nd being a value returned by
                getvalue.
            value (callable, optional): if given, call on the object
                in question, and return a value that will be passed to
                condition. Then return value in addition to the object
                if condition is True.

        Returns:
            Node, or (Node, value) if getvalue is given
        """
        visited_nodes = {}

        # import inspect
        # try:
        #     _s = inspect.stack()[3][3]
        # except IndexError:
        #     _s = inspect.stack()[1][3]
        # print("== parent_bfs", _s)

        if getvalue is None:
            # print("..parent_bfs:", self, condition(self, None))
            if condition(self, None):
                return self
        else:
            val = getvalue(self)
            # print("..parent_bfs:", self, condition(self, val), val)
            if condition(self, val):
                return self, val

        this_level = []
        next_level = self.parents
        while len(next_level) > 0:
            this_level = next_level
            next_level = []
            for obj in this_level:
                # check for circular refernces
                if id(obj) in visited_nodes:
                    continue
                # see if this is our node, and return if true
                if getvalue is None:
                    # print("..parent_bfs:", obj, condition(obj, None))
                    if condition(obj, None):
                        return obj
                else:
                    value = getvalue(obj)
                    # print("..parent_bfs:", obj, condition(obj, value), value)
                    if condition(obj, value):
                        return obj, value
                # setup the next level parent-ward
                visited_nodes[id(obj)] = True
                next_level += obj.parents

        if getvalue is None:
            return None
        else:
            return None, None

    def has_info(self, key):
        return key in self._info

    def get_info(self, key):
        return self._info[key]

    def get_all_info(self):
        all_info = dict()
        def condition(obj, val):  # pylint: disable=unused-argument
            for k, v in obj._info.items():  # pylint: disable=protected-access
                if not k in all_info:
                    all_info[k] = v
            return False
        self._parent_bfs(condition)
        return all_info

    def print_info_tree(self):
        lines = []
        def stringifiy_items(obj, val):  # pylint: disable=unused-argument
            prefix = "...."
            lines.append(prefix + ":: " + str(obj))
            for k, v in obj._info.items():  # pylint: disable=protected-access
                lines.append(prefix + "  " + str(k) + " = " + str(v))
            return False
        self._parent_bfs(stringifiy_items)
        print("\n".join(lines))

    def set_info(self, key, val):
        self._info[key] = val

    def find_info_owner(self, key):
        """Go through the parents (breadth first) and find the info

        Raises:
            KeyError
        """
        condition = lambda obj, val: obj.has_info(key)
        matching_parent = self._parent_bfs(condition)
        if matching_parent is None:
            raise KeyError("info '{0}' is nowhere to be found".format(key))
        else:
            return matching_parent

    def find_info(self, key):
        """Go through the parents (breadth first) and find the info

        Raises:
            KeyError
        """
        matching_parent = self.find_info_owner(key)
        return matching_parent.get_info(key)

    def update_info(self, key, val, fallback=True):
        """Update an existing key if found, or fall back to add_info"""
        try:
            matching_parent = self.find_info_owner(key)
            matching_parent.set_info(key, val)
        except KeyError:
            if fallback:
                self.set_info(key, val)
            else:
                raise

    ##########################
    # for time related things

    # these _sub_* methods are intended to be overridden
    def _sub_translate_time(self, time):  # pylint: disable=no-self-use
        return NotImplemented

    def _sub_format_time(self, time, style):  # pylint: disable=unused-argument,no-self-use
        return NotImplemented

    def _sub_time_as_datetime(self, time, epoch):  # pylint: disable=no-self-use
        return NotImplemented

    # these routines should not be overridden
    def _translate_time(self, time):
        """Translate time from one representation to a float

        Note:
            do not override this function, instead override
            _sub_translate_time since that is automagically
            monkey-patched along the tree of datasets / grids / fields

        Returns:
            NotImplemented or float representation of time for the
            current dataset
        """
        getvalue = lambda obj: obj._sub_translate_time(time)  # pylint: disable=protected-access
        condition = lambda obj, val: val != NotImplemented
        _, val = self._parent_bfs(condition, getvalue)  # pylint: disable=unpacking-non-sequence,unbalanced-tuple-unpacking
        if val is not None:
            return val

        # parse a string, if that's what time is
        if isinstance(time, string_types):
            time = time.lstrip(ascii_letters)
            # there MUST be a better way to do this than 3 nested trys
            try:
                time = datetime.strptime(time, "%H:%M:%S.%f")
            except ValueError:
                try:
                    time = datetime.strptime(time, "%d:%H:%M:%S.%f")
                except ValueError:
                    try:
                        time = datetime.strptime(time, "%m:%d:%H:%M:%S.%f")
                    except ValueError:
                        pass

        # figure out the datetime, if that's what time is
        if isinstance(time, datetime):
            delta = time - datetime.strptime("00", "%S")
            return delta.total_seconds()

        return NotImplemented

    def format_time(self, style=".02f", time=None):
        """Format time as a string

        See Also:
            :py:func:`viscid.vutil.format_time`

        Returns:
            string
        """
        if time is None:
            time = self.time

        getvalue = lambda obj: obj._sub_format_time(time, style)  # pylint: disable=protected-access
        condition = lambda obj, val: val != NotImplemented
        _, val = self._parent_bfs(condition, getvalue)  # pylint: disable=unpacking-non-sequence,unbalanced-tuple-unpacking
        if val is not None:
            return val
        return generic_format_time(time, style)

    def time_as_datetime(self, time=None, epoch=None):
        """Convert floating point time to datetime

        Args:
            t (float): time
            epoch (datetime): if None, uses Jan 1 1970
        """
        getvalue = lambda obj: obj._sub_time_as_datetime(self.time, epoch)  # pylint: disable=protected-access
        condition = lambda obj, val: val != NotImplemented
        _, val = self._parent_bfs(condition, getvalue)  # pylint: disable=unpacking-non-sequence,unbalanced-tuple-unpacking
        if val is not None:
            return val

        dt = self.time_as_timedelta(self.time)
        if epoch is None:
            epoch = datetime.utcfromtimestamp(0)
        return epoch + dt

    def time_as_timedelta(self, time=None):  # pylint: disable=no-self-use
        """Convert floating point time to a timedelta"""
        if time is None:
            time = self.time
        return timedelta(seconds=time)


class Leaf(Node):
    """Base class for fields"""
    pass

##
## EOF
##
