# If we have lxml, use it, else use a modification of the Python std lib xml

from viscid import logger

force_native_xml = True

try:
    if force_native_xml:
        raise ImportError

    from lxml import etree
    logger.debug("Using lxml library")

    def parse(fname, **kwargs):
        return etree.parse(fname, **kwargs)

    def xinclude(tree, base_url=None, **kwargs):
        """Summary

        Args:
            tree (Tree): The object returned by parse
            base_url (str): Not used
            **kwargs: passed to tree.xinclude()
        """
        # TODO: ignore if an xincluded xdmf file doesn't exist?
        if base_url:
            logger.warning("lxml will ignore base_url: %s", base_url)
        return tree.xinclude(**kwargs)

except ImportError:
    from xml.etree import ElementTree

    from viscid.compat import _xdmf_include

    logger.debug("Using native xml library")

    def parse(fname, **kwargs):
        return ElementTree.parse(fname, **kwargs)

    def xinclude(tree, base_url=None, **kwargs):
        """Summary

        Args:
            tree (Tree): The object returned by parse
            base_url (str): Interpret xinclude paths relative to this
            **kwargs: passed to _xdmf_include.include
        """
        root = tree.getroot()
        _xdmf_include.include(root, base_url=base_url, **kwargs)
