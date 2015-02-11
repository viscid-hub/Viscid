#!/usr/bin/env python

from __future__ import print_function
import argparse

from viscid import logger
from viscid import readers
from viscid import vutil

def main():
    parser = argparse.ArgumentParser(description="Load some data files")
    parser.add_argument('files', nargs="*", help='input files')
    args = vutil.common_argparse(parser)

    # print("args", args)
    # logger.error(args)
    # logger.warn(args)
    logger.info("args: {0}".format(args))
    # logger.debug(args)
    print()

    files = readers.load_files(args.files)
    readers.__filebucket__.print_tree()
    print()

    for f in files:
        f.print_tree(depth=-1)

if __name__ == "__main__":
    main()

##
## EOF
##
