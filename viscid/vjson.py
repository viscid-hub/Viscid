"""Wrap python's json parser enabling comments and trailing commas

Comments are either # or //
"""
from __future__ import print_function

import sys

try:
    import json as _json
except ImportError:
    # If python version is 2.5 or less, use simplejson
    import simplejson as _json  # pylint: disable=import-error

def find_all(s, substrings, start=0, end=None):
    """Iterate all ocurances of sub in s

    Args:
        s (str): whole string
        sub (str, list): substring or list of substrings to find

    Yields:
        index of start of sub in s
    """
    if not isinstance(substrings, (list, tuple)):
        substrings = [substrings]
    while start >= 0:
        min_found = -1
        sublen = 0
        for sub in substrings:
            ind = s.find(sub, start, end)
            if ind >= 0 and (ind < min_found or min_found == -1):
                min_found = ind
                sublen = len(sub)
        start = min_found
        if start == -1:
            return
        yield start
        start += sublen  # use start += 1 to find overlapping matches

def rfind_all(s, substrings, start=0, end=None):
    """Iterate all ocurances of sub in s

    Args:
        s (str): whole string
        sub (str, list): substring or list of substrings to find

    Yields:
        index of start of sub in s
    """
    if not isinstance(substrings, (list, tuple)):
        substrings = [substrings]
    while start >= 0:
        max_found = -1
        sublen = 0
        for sub in substrings:
            ind = s.rfind(sub, start, end)
            if ind > max_found:
                max_found = ind
                sublen = len(sub)
        end = max_found
        if end == -1:
            return
        yield end
        end -= (sublen - 1)  # use 0 for overlapping matches

def loads(text, **kwargs):
    """Wrap Python's :py:func:`json.loads`

    Trims out comments and trailing commas first

    Args:
        text: json as a string
        **kwargs: passed to Python's :py:func:`json.loads`
    """
    lines = text.split('\n')

    last_nonempty = -1
    for i in range(len(lines)):
        # remove comments
        for loc in find_all(lines[i], ('#', '//')):
            # if there are an odd number of single or double quotes, then
            # the comment char is part of a string, and should be kept
            if (lines[i].count('"', 0, loc) % 2 == 0 and
                lines[i].count("'", 0, loc) % 2 == 0):  # pylint: disable=bad-continuation
                lines[i] = lines[i][:loc]
                break

        lines[i] = lines[i].rstrip()

        # trailing comma?
        if lines[i]:
            remove_at = []
            for loc in find_all(lines[i], ('}', ']')):
                preceeding_txt = lines[i][:loc].rstrip()
                # if no preceeding_txt on same line, check last_nonwhite
                if not preceeding_txt:
                    if last_nonempty > -1:
                        if lines[last_nonempty][-1] == ',':
                            lines[last_nonempty] = lines[last_nonempty][:-1]
                else:
                    if preceeding_txt[-1] == ',':
                        remove_at.append(len(preceeding_txt) - 1)
            for j, loc in enumerate(remove_at):
                loc -= j
                lines[i] = lines[i][:loc] + lines[i][loc + 1:]

            if lines[i]:
                last_nonempty = i

    return _json.loads('\n'.join(lines), **kwargs)

def load(f, **kwargs):
    """Read file and pass to :py:func:`loads`

    Args:
        f: file-like object with a read method
        **kwargs: passed to :py:func:`loads`
    """
    try:
        return loads(f.read(), **kwargs)
    except Exception:
        print("JSON Error on file: {0}".format(f.name), file=sys.stderr)
        raise

def dumps(obj, **kwargs):
    """Wrap Python's :py:func:`json.dumps`

    Args:
        obj: some object that Python's :py:func:`json.dumps` understands
        **kwargs: passed to Python's :py:func:`json.dumps`
    """
    return _json.dumps(obj, **kwargs)

def dump(obj, f, **kwargs):
    """Wrap Python's :py:func:`json.dump`

    Args:
        obj: some object that Python's :py:func:`json.dump` understands
        f: file-like object with a write method
        **kwargs: passed to Python's :py:func:`json.dump`
    """
    return _json.dump(obj, f, **kwargs)

##
## EOF
##
