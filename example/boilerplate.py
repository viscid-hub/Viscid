#!/usr/bin/env python

from __future__ import print_function
import argparse
import logging

from viscid import readers
from viscid import vutil

def main():
    parser = argparse.ArgumentParser(description="Load some data files")
    parser.add_argument('files', nargs="*", help='input files')
    args = vutil.common_argparse(parser)

    print("args", args)
    # logging.error(args)
    # logging.warn(args)
    # logging.info(args)
    # logging.debug(args)
    print()

    files = readers.load(args.files)
    readers.__filebucket__.spill()
    print()
    
    for f in files:
        f.spill(recursive=True)

if __name__ == "__main__":
    main()

##
## EOF
##
