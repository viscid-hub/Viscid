#!/usr/bin/env python

from __future__ import print_function
import json
import os
import shutil
import sys


INSTALL_MANIFEST = '_install_manifest.json'


def _main():
    verb = '-v' in sys.argv or '--verbose' in sys.argv
    verb = True

    # read the install manifest
    with open(INSTALL_MANIFEST, 'r') as fin:
        inst_manifest = json.load(fin)

    if sys.executable in inst_manifest:
        # remove the whole package directory, which now should just
        # be populated with empty subdirectories
        file_list = inst_manifest[sys.executable]['file_list']
        for fname in file_list:
            if verb:
                print("Removing:", fname, file=sys.stderr)

            try:
                os.remove(fname)
            except FileNotFoundError:
                pass

        # remove the whole package directory, which now should just
        # be populated with empty subdirectories
        try:
            pkg_instdir = inst_manifest[sys.executable]['pkg_instdir']

            if verb:
                print("Remove tree:", pkg_instdir, file=sys.stderr)
            shutil.rmtree(pkg_instdir, ignore_errors=False)
        except OSError:
            pass

        # pretend we were never installed in this and rewrite the
        # install manifest
        del inst_manifest[sys.executable]

        with open(INSTALL_MANIFEST, 'w') as fout:
            json.dump(inst_manifest, fout, indent=2, sort_keys=True)
    elif verb:
        print("Uninstall: not in manifest for", sys.executable,
              file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(_main())

##
## EOF
##
