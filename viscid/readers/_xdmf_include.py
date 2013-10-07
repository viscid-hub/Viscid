# This is a take on the xinclude mechanism in the element tree implementation
# that comes with Python. The difference is that in this implementation, href
# paths are relative to the xdmf file, not the cwd. Also, there is some very
# limited support for XPointers.

import copy
import code
import logging
from xml.etree import ElementTree

XINCLUDE = "{http://www.w3.org/2001/XInclude}"

XINCLUDE_INCLUDE = XINCLUDE + "include"
XINCLUDE_FALLBACK = XINCLUDE + "fallback"

class FatalIncludeError(SyntaxError):
    pass

def default_loader(href, parse, xi_elem, encoding=None):
    # code.interact(local=locals())
    try:
        if parse == "xml":
            data = ElementTree.parse(href).getroot()
        else:
            with open(href) as f:
                data = f.read()
                if encoding:
                    data = data.decode(encoding)
    except IOError:
        # as far as I care, if a file doesn't exist, that's ok
        data = None
    return data

def include(elem, loader=None):
    if loader is None:
        loader = default_loader
    # look for xinclude elements
    i = 0
    while i < len(elem):
        e = elem[i]
        if e.tag == XINCLUDE_INCLUDE:
            # process xinclude directive
            href = e.get("href")
            parse = e.get("parse", "xml")
            pointer = e.get("xpointer", None)

            if parse == "xml":
                node = loader(href, parse, e)

                if node is None:
                    logging.debug("XInclude: File '{0}' not found".format(href))
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
            include(e, loader)
        i = i + 1

##
## EOF
##
