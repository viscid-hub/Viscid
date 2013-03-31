#!/usr/bin/env python

from __future__ import print_function
import argparse

from viscid import readers

verb = 0

def main():
    parser = argparse.ArgumentParser(description="Load some data files")
    parser.add_argument('-v', action='count', default=0,
                        help='increase verbosity')
    parser.add_argument('-q', action='count', default=0,
                        help='decrease verbosity')
    parser.add_argument('files', nargs="*", help='input files')
    args = parser.parse_args()
    verb = args.v - args.q

    print(args)

    files = readers.load(args.files)


if __name__ == "__main__":
    main()

##
## EOF
##
