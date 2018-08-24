"""Classes for elements along the tree from files to fields"""

from __future__ import print_function

import viscid


class _NO_DEFAULT_GIVEN(object):
    pass


class Node(object):
    """Base class for Datasets and Grids"""

    name = None
    _info = None
    parents = None

    def __init__(self, name=None, time=None, info=None, parents=None, **kwargs):
        if name is None:
            name = "<{0} @ {1}>".format(self.__class__.__name__, hex(id(self)))
        self.name = name

        if info is None:
            info = dict()
        self._info = info

        self.time = time

        if parents is None:
            parents = []
        if not isinstance(parents, (list, tuple)):
            parents = [parents]
        self.parents = parents

        self._info.update(kwargs)

    def prepare_child(self, obj):
        if self not in obj.parents:
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

    def get_info(self, key, default=_NO_DEFAULT_GIVEN):
        try:
            return self._info[key]
        except KeyError:
            if default is _NO_DEFAULT_GIVEN:
                raise
            else:
                return default

    def get_all_info(self):
        all_info = dict()
        def condition(obj, val):  # pylint: disable=unused-argument
            for k, v in obj._info.items():  # pylint: disable=protected-access
                if k not in all_info:
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
        def condition(obj, _):
            return obj.has_info(key)
        matching_parent = self._parent_bfs(condition)
        if matching_parent is None:
            raise KeyError("info '{0}' is nowhere to be found".format(key))
        else:
            return matching_parent

    def find_info(self, key, default=_NO_DEFAULT_GIVEN):
        """Go through the parents (breadth first) and find the info"""
        try:
            matching_parent = self.find_info_owner(key)
            return matching_parent.get_info(key)
        except KeyError:
            if default is _NO_DEFAULT_GIVEN:
                raise
            else:
                return default

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

    def find_attr(self, attr_name, default=_NO_DEFAULT_GIVEN):
        """Breadth first search of parents for attr_name

        Args:
            attr_name (str): some attribute name
            default (Any): fallback, possibly raises AttributeError
                if this is not given

        Raises:
            AttributeError: if no default given, and no parent found
                with attr_name

        Returns:
            The attribute, or default
        """
        def _condition(_node, _):
            return (hasattr(_node, attr_name) and
                    getattr(parent, attr_name) is not NotImplemented)
        parent = self._parent_bfs(_condition)
        if parent is not None:
            return getattr(parent, attr_name)
        else:
            if default is _NO_DEFAULT_GIVEN:
                raise AttributeError("No parent of {0} has an attribute {1}"
                                     "".format(self, attr_name))
            else:
                return default

    ##########################
    # for time related things

    @property
    def basetime(self):
        basetime = self.find_info('basetime', default=None)
        if basetime is None:
            s = ("Node {0} did not set basetime, so it can't deal with "
                 "datetimes. Maybe you forgot to set basetime, or your "
                 "logfile is missing or mangled.".format(repr(self)))
            raise viscid.NoBasetimeError(s)
        return viscid.as_datetime64(basetime)

    @basetime.setter
    def basetime(self, val):
        self.set_info('basetime', viscid.as_datetime64(val))

    @property
    def time(self):
        return self.find_info('time', default=None)

    @time.setter
    def time(self, val):
        if viscid.is_datetime_like(val):
            val = viscid.as_timedelta(self.basetime - viscid.as_datetime64(val))
            val = val.total_seconds()
        elif viscid.is_timedelta_like(val, conservative=True):
            val = viscid.as_timedelta(val).total_seconds()
        elif val is not None:
            self.set_info('time', float(val))

    def time_as_timedelta64(self):
        return viscid.as_timedelta64(1e6 * self.time, 'us')

    def time_as_timedelta(self):
        return viscid.as_timedelta(self.time_as_timedelta64())

    def time_as_datetime64(self):
        return self.t2datetime64(self.time_as_timedelta64())

    def time_as_datetime(self):
        return viscid.as_datetime(self.time_as_datetime64())

    def t2datetime64(self, t):
        return viscid.time_sum(self.basetime, t)

    def format_time(self, fmt='.02f', default="Timeless"):
        if self.time is None:
            return default
        else:
            try:
                return viscid.format_time(self.time_as_datetime64(), fmt=fmt,
                                          basetime=self.basetime)
            except viscid.NoBasetimeError:
                return viscid.format_time(self.time_as_timedelta64(), fmt=fmt)

    def resolve(self):
        return self

    def iter_resolved_children(self):
        return (child.resolve() for child in self.children)


class Leaf(Node):
    """Base class for fields"""
    pass

##
## EOF
##
