# This is a take on the xinclude mechanism in the element tree implementation
# that comes with Python. The difference is that in this implementation, href
# paths are relative to the xdmf file, not the cwd. Also, there is some very
# limited support for XPointers.

import copy
from xml.etree import ElementTree
try:
    from urlparse import urljoin  # pylint: disable=E0611
    from urllib2 import urlopen  # pylint: disable=E0611
except ImportError:
    # Python 3
    from urllib.parse import urljoin  # pylint: disable=E0611,F0401
    from urllib.request import urlopen  # pylint: disable=E0611,F0401

try:
    set
except NameError:
    # Python 2.3
    from sets import Set as set  # pylint: disable=W0622

from viscid import logger

XINCLUDE = "{http://www.w3.org/2001/XInclude}"

XINCLUDE_INCLUDE = XINCLUDE + "include"
XINCLUDE_FALLBACK = XINCLUDE + "fallback"

class FatalIncludeError(SyntaxError):
    pass

def _xdmf_default_loader(href, parse, encoding=None, parser=None):
    try:
        if parse == "xml":
            data = ElementTree.parse(href, parser).getroot()
        else:
            if "://" in href:
                with urlopen(href) as f:
                    data = f.read()
            else:
                with open(href, 'rb') as f:
                    data = f.read()

            if not encoding:
                encoding = 'utf-8'
            data = data.decode(encoding)
    except IOError:
        # as far as I care, if a file doesn't exist, that's ok
        data = None
    return data

def include(elem, loader=None, base_url="./", _parent_hrefs=None):
    """ base_url is just a file path """
    if loader is None:
        loader = _xdmf_default_loader

    # TODO: for some nested includes, urljoin is adding an extra / which
    # means this way of detecting infinite recursion doesn't work
    if _parent_hrefs is None:
        _parent_hrefs = set()

    # look for xinclude elements
    i = 0
    while i < len(elem):
        e = elem[i]
        if e.tag == XINCLUDE_INCLUDE:
            # process xinclude directive
            href = e.get("href")
            href = urljoin(base_url, href)
            parse = e.get("parse", "xml")
            pointer = e.get("xpointer", None)

            if parse == "xml":
                if href in _parent_hrefs:
                    raise FatalIncludeError(
                        "recursive include of {0} detected".format(href)
                        )
                _parent_hrefs.add(href)
                node = loader(href, parse, e)

                if node is None:
                    logger.debug("XInclude: File '{0}' not found".format(href))
                    del elem[i]
                    continue

                # trying to use our limited xpointer / xpath support?
                if pointer is not None:
                    # really poor man's way of working around the fact that
                    # default etree can't do absolute xpaths
                    if pointer.startswith("xpointer("):
                        pointer = pointer[9:-1]
                    if pointer.startswith(node.tag):
                        pointer = "./" + "/".join(pointer.split("/")[1:])
                    if pointer.startswith("/" + node.tag):
                        pointer = "./" + "/".join(pointer.split("/")[2:])
                    if pointer.startswith("//" + node.tag):
                        pointer = ".//" + "/".join(pointer.split("/")[3:])
                    node = copy.copy(node.find(pointer))
                else:
                    node = copy.copy(node)

                # recursively look for xincludes in the included element
                include(node, loader, href, _parent_hrefs)

                if e.tail:
                    node.tail = (node.tail or "") + e.tail
                elem[i] = node

            elif parse == "text":
                text = loader(href, parse, e, e.get("encoding"))
                if text is None:
                    raise FatalIncludeError(
                        "cannot load %r as %r" % (href, parse)
                        )
                if i:
                    node = elem[i-1]
                    node.tail = (node.tail or "") + text
                else:
                    elem.text = (elem.text or "") + text + (e.tail or "")
                del elem[i]
                continue
            else:
                raise FatalIncludeError(
                    "unknown parse type in xi:include tag (%r)" % parse
                )
        else:
            include(e, loader, base_url, _parent_hrefs)
        i = i + 1

##
## EOF
##
