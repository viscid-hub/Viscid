#!/usr/bin/env python
"""Edge and face centered tools"""

from __future__ import division, print_function

try:
    import numexpr as ne
    _HAS_NUMEXPR = True
except ImportError:
    _HAS_NUMEXPR = False
import numpy as np

import viscid


__all__ = ['STAGGER_LEADING', 'STAGGER_TRAILING', 'PREPROCESSED_KEY', 'FLIP_KEY',
           'STAGGER_KEY', 'make_ecfc_field_leading', 'fc2cc', 'ec2cc', 'div_fc']

# The leading / trailing nomenclature says how nc/cc indices are
# staggered, ie, for leading, cc_i = 0.5 * (fc_i + fc_i+1). This is
# the case for the C3 stepper, etc. Trailing is the convention used
# by the fortran kernel, ie, cc_i = 0.5 * (fc_i-1 + fc_i)

STAGGER_LEADING = 0
STAGGER_TRAILING = 1

PREPROCESSED_KEY = "ecfc_preprocessed"
FLIP_KEY = "ecfc_flip"
STAGGER_KEY = "ecfc_staggering"


def _prep_slices(f):
    is_leading = f.find_info(PREPROCESSED_KEY, False)
    is_leading |= f.find_info(STAGGER_KEY, None) == [STAGGER_LEADING] * 3
    if not is_leading:
        raise RuntimeError("Please make f leading staggered first")

    _s0 = slice(None, -1)
    _smL = slice(None, -1)
    _spL = slice(1, None)

    s0 = [None] * 3
    sm = [None] * 3
    sp = [None] * 3

    for i, n in enumerate(f.sshape):
        if n == 1:
            s0[i], sm[i], sp[i] = slice(None), slice(None), slice(None)
        elif n >= 2:
            s0[i], sm[i], sp[i] = _s0, _smL, _spL
        else:
            raise RuntimeError("I shouldn't be here")
    return s0, sm, sp

def make_ecfc_field_leading(fc, trim_leading=True):
    """Standardize staggering on edge / face centered fields"""
    if fc.find_info(PREPROCESSED_KEY, False):
        return fc
    elif fc.find_info(PREPROCESSED_KEY, None) is None:
        # this is for non-ggcm fields
        fc.set_info(PREPROCESSED_KEY, True)
        fc.set_info(STAGGER_KEY, [STAGGER_LEADING] * 3)
        return fc
    else:
        # NOTE: I deeply apologize for the variable names and logic here,
        # it turns out to be super confusing to deal with staggering and
        # flipping of 2 components (mhd->gse), etc. The important thing is
        # this logic was tested to work with step_c, step_c3, and jrrle
        # output in both mhd and gse crds
        center = fc.center.lower()
        if center not in ('face', 'edge'):
            raise ValueError("This only makes sense for Edge / Face centered "
                             "fields")

        SA = [slice(None, -1), slice(1, None), slice(None, None)]
        SB = [SA[1], SA[0], SA[2]]

        # Trailing offsets/Staggering
        _sd0T = 1
        _sdT = 0
        # Leading Offsets/Staggering
        if trim_leading:
            # this makes everything symmetric since trailing offset fields
            # will be sliced by one value, so you can slice the first and
            # last values from a CC field and add it to an fc2cc field
            # regardless of the fc field's representation.
            _sd0L = 1
            _sdL = 1
        else:
            _sd0L = 2
            _sdL = 2

        # sd: how to slice the data for vector component xyz
        # sc: how to slice the xyz NC coordinates
        sd = [[None] * 3, [None] * 3, [None] * 3]
        sc = [None] * 3
        # staggering = fc.find_info(STAGGER_KEY, [STAGGER_TRAILING] * 3)
        staggering = fc.find_info(STAGGER_KEY, [STAGGER_LEADING] * 3)
        flipping = fc.find_info(FLIP_KEY, [False] * 3)

        for i, n in enumerate(fc.sshape):
            if staggering[i] == STAGGER_TRAILING:
                _sd0, _sd = _sd0T, _sdT
            else:
                _sd0, _sd = _sd0L, _sdL

            if n == 1:
                sc[i] = slice(None, None)
                for j in range(3):
                    sd[i][j] = slice(None, None)
            elif n > 1:
                sc[i] = slice(1, None)
                for j in range(3):
                    cnd = i == j if center == 'face' else i != j
                    S = SB if flipping[j] and cnd else SA
                    if cnd:
                        sd[i][j] = S[_sd]
                    else:
                        sd[i][j] = S[_sd0]
            else:
                raise RuntimeError("I shouldn't be here")

        fc.resolve()  # if translating -> gse, do it once, not 4 times
        s0vec = list(sc)
        s0vec.insert(fc.nr_comp, slice(None))
        prepared_fc = viscid.zeros_like(fc[s0vec])

        prepared_fc['x'] = fc['x', sd[0][0], sd[0][1], sd[0][2]]
        prepared_fc['y'] = fc['y', sd[1][0], sd[1][1], sd[1][2]]
        prepared_fc['z'] = fc['z', sd[2][0], sd[2][1], sd[2][2]]

        prepared_fc.set_info(PREPROCESSED_KEY, True)
        prepared_fc.set_info(STAGGER_KEY, [STAGGER_LEADING] * 3)
        prepared_fc.name = fc.name
        prepared_fc.pretty_name = fc.pretty_name
        return prepared_fc

def fc2cc(fc, force_numpy=False, bnd=True):
    """Average a face centered field to cell centers"""
    fc = fc.atleast_3d()
    fc = make_ecfc_field_leading(fc)
    s0, sm, sp = _prep_slices(fc)

    s0vec = list(s0)
    s0vec.insert(fc.nr_comp, slice(None))
    cc = viscid.zeros_like(fc[s0vec])
    cc.center = "Cell"

    fcx0, fcx1 = (fc['x', sm[0], s0[1], s0[2]].data,
                  fc['x', sp[0], s0[1], s0[2]].data)
    fcy0, fcy1 = (fc['y', s0[0], sm[1], s0[2]].data,
                  fc['y', s0[0], sp[1], s0[2]].data)
    fcz0, fcz1 = (fc['z', s0[0], s0[1], sm[2]].data,
                  fc['z', s0[0], s0[1], sp[2]].data)

    if _HAS_NUMEXPR and not force_numpy:
        half = np.array([0.5], dtype=fc.dtype)[0]  # pylint: disable=unused-variable
        s = "half * (a + b)"
        a, b = fcx0, fcx1  # pylint: disable=unused-variable
        cc['x'] = ne.evaluate(s)
        a, b = fcy0, fcy1
        cc['y'] = ne.evaluate(s)
        a, b = fcz0, fcz1
        cc['z'] = ne.evaluate(s)
    else:
        cc['x'] = 0.5 * (fcx0 + fcx1)
        cc['y'] = 0.5 * (fcy0 + fcy1)
        cc['z'] = 0.5 * (fcz0 + fcz1)

    if bnd:
        # FIXME: this is really just faking the bnd so there aren't shape
        #        errors when doing math with the result
        cc = viscid.extend_boundaries(cc, nl=1, nh=1, order=0, crd_order=1)
    cc.name = fc.name
    cc.pretty_name = fc.pretty_name
    return cc

def ec2cc(ec, force_numpy=False, bnd=True):
    """Average an edge centered field to cell centers"""
    ec = ec.atleast_3d()
    ec = make_ecfc_field_leading(ec)
    s0, sm, sp = _prep_slices(ec)

    s0vec = list(s0)
    s0vec.insert(ec.nr_comp, slice(None))
    cc = viscid.zeros_like(ec[s0vec])
    cc.center = "Cell"

    ecx0, ecx1, ecx2, ecx3 = (ec['x', s0[0], sm[1], sm[2]].data,
                              ec['x', s0[0], sm[1], sp[2]].data,
                              ec['x', s0[0], sp[1], sm[2]].data,
                              ec['x', s0[0], sp[1], sp[2]].data)
    ecy0, ecy1, ecy2, ecy3 = (ec['y', sm[0], s0[1], sm[2]].data,
                              ec['y', sm[0], s0[1], sp[2]].data,
                              ec['y', sp[0], s0[1], sm[2]].data,
                              ec['y', sp[0], s0[1], sp[2]].data)
    ecz0, ecz1, ecz2, ecz3 = (ec['z', sm[0], sm[1], s0[2]].data,
                              ec['z', sm[0], sp[1], s0[2]].data,
                              ec['z', sp[0], sm[1], s0[2]].data,
                              ec['z', sp[0], sp[1], s0[2]].data)

    if _HAS_NUMEXPR and not force_numpy:
        quarter = np.array([0.25], dtype=ec.dtype)[0]  # pylint: disable=unused-variable
        s = "quarter * (a + b + c + d)"
        a, b, c, d = ecx0, ecx1, ecx2, ecx3  # pylint: disable=unused-variable
        cc['x'] = ne.evaluate(s)
        a, b, c, d = ecy0, ecy1, ecy2, ecy3
        cc['y'] = ne.evaluate(s)
        a, b, c, d = ecz0, ecz1, ecz2, ecz3
        cc['z'] = ne.evaluate(s)
    else:
        cc['x'] = 0.25 * (ecx0 + ecx1 + ecx2 + ecx3)
        cc['y'] = 0.25 * (ecy0 + ecy1 + ecy2 + ecy3)
        cc['z'] = 0.25 * (ecz0 + ecz1 + ecz2 + ecz3)

    if bnd:
        # FIXME: this is really just faking the bnd so there aren't shape
        #        errors when doing math with the result
        cc = viscid.extend_boundaries(cc, nl=1, nh=1, order=0, crd_order=1)
    cc.name = ec.name
    cc.pretty_name = ec.pretty_name
    return cc


def div_fc(fc, force_numpy=False, bnd=True):
    """Calculate cell centered divergence of face centered field"""
    fc = fc.atleast_3d()
    fc = make_ecfc_field_leading(fc)
    # FIXME: maybe it's possible to do the boundary correctly here without
    #        just faking it with a 0 order hold before returning
    s0, sm, sp = _prep_slices(fc)

    s0vec = list(s0)
    s0vec.insert(fc.nr_comp, slice(0, 1))
    div_cc = viscid.zeros_like(fc[s0vec], center="cell")

    x, y, z = fc.get_crds_nc('xyz', shaped=True)
    if True:
        # x, y, z = x[1:, :, :], y[:, 1:, :], z[:, :, 1:]
        x, y, z = x[:-1, :, :], y[:, :-1, :], z[:, :, :-1]
    else:
        raise NotImplementedError()
    # x, y, z = fc.get_crds_cc('xyz', shaped=True)

    xm, xp = x[sm[0], :, :], x[sp[0], :, :]
    ym, yp = y[:, sm[1], :], y[:, sp[1], :]
    zm, zp = z[:, :, sm[2]], z[:, :, sp[2]]

    fcx0, fcx1 = (fc['x', sm[0], s0[1], s0[2]].data,
                  fc['x', sp[0], s0[1], s0[2]].data)
    fcy0, fcy1 = (fc['y', s0[0], sm[1], s0[2]].data,
                  fc['y', s0[0], sp[1], s0[2]].data)
    fcz0, fcz1 = (fc['z', s0[0], s0[1], sm[2]].data,
                  fc['z', s0[0], s0[1], sp[2]].data)

    # xp, yp, zp = xm + 1.0, ym + 1.0, zm + 1.0

    if _HAS_NUMEXPR and not force_numpy:
        div_cc[:, :, :] = ne.evaluate("((fcx1 - fcx0) / (xp - xm)) + "
                                      "((fcy1 - fcy0) / (yp - ym)) + "
                                      "((fcz1 - fcz0) / (zp - zm))")
    else:
        div_cc[:, :, :] = (((fcx1 - fcx0) / (xp - xm)) +
                           ((fcy1 - fcy0) / (yp - ym)) +
                           ((fcz1 - fcz0) / (zp - zm)))

    if bnd:
        # FIXME: this is really just faking the bnd so there aren't shape
        #        errors when doing math with the result
        div_cc = viscid.extend_boundaries(div_cc, nl=1, nh=1, order=0,
                                          crd_order=1)
    div_cc.name = "div " + fc.name
    div_cc.pretty_name = "Div " + fc.pretty_name
    return div_cc

##
## EOF
##
