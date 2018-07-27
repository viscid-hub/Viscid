#!/usr/bin/env python

from __future__ import print_function

import sympy

import viscid


def _main():
    print("Matrices applied in the order given "
          "(i.e., matmul is in reversed order)")
    print()
    for order in ('zyx',):
        print(order.upper(), ': ',
              ' * '.join(ax.upper() + str(i)
                         for i, ax in zip(range(2, -1, -1), order[::-1])),
              sep='')
        print('-----------------')
        sympy.pprint(viscid.symbolic_rot(axes=order))

if __name__ == "__main__":
    _main()

##
## EOF
##
