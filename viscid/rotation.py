#!/usr/bin/env bash
"""Euler angle <-> Rotation matrix <-> quaternion conversions

This module is designed to work with Numpy versions 1.9+

Quaternion functions will work with vanilla Numpy, but they can also
make use of `this quaternion library <https://github.com/moble/quaternion>`_.
It is probably to your benefit to use a library that explicitly
implements quaternion algebra.

Note:
    If a function in this module has the same name as one in Matlab,
    the conventions used for the functions are the same.

Conventions:
    Rotation matrices and Euler angles are notoriously ambiguous due to
    competing conventions for order and meaning. Hopefully this section
    makes it clear which conventions are used in which functions.

    - **What is rotating**
        Rotations always follow the right hand rule, but can mean two
        things depending what is rotating.

        * **Extrinsic rotations** ``*rot*``
            Axes remain fixed, and matrices rotate vectors. This
            convention is used in functions with ``rot`` in their names
            to mirror Matlab's Robotics System toolkit. For example::

            >>> e_rot(numpy.pi / 2, 'z') @ [1, 0, 0] = [0, 1, 0]

        * **Intrinsic rotations** ``*dcm*``
            Vectors remain fixed, but matrices rotate axes. This
            convention is used in functions with ``dcm`` in their names
            to mirror Matlab's Aerospace toolkit. For example::

            >>> e_dcm(numpy.pi / 2, 'z') @ [1, 0, 0] = [0, -1, 0]

    - **Axis Order**
        For functions that deal with Euler angles, their order can be
        specified in two ways

        * **Multiplication order** ``*eul*``
            Axes are specified in the same order as the multiplication
            of the elementary matrices. In other words, the last axis
            specified is the first rotation applied. This convention is
            used in functions with ``eul`` in their names to mirror
            Matlab's Robotics System toolkit.

        * **Transformation order** ``*angle*``
            Axes are specified in the order that they are applied. In
            other words, the first axis is the first rotation, or the
            right-most matrix. This convention is used in functions with
            ``angle`` in their names to mirror Matlab's Aerospace toolkit.

Functions:
    - angle2rot: Euler angles (transform-order) -> rotation matrix (extrinsic)
    - eul2rot: Euler angles (multiplication-order) -> rotation matrix (extrinsic)
    - angle2dcm: Euler angles (transform-order) -> rotation matrix (intrinsic)
    - eul2dcm: Euler angles (multiplication-order) -> rotation matrix (intrinsic)

    - rot2angle: Rotation matrix (extrinsic) -> Euler angles (transform-order)
    - rot2eul: Rotation matrix (extrinsic) -> Euler angles (multiplication-order)
    - dcm2angle: Rotation matrix (intrinsic) -> Euler angles (transform-order)
    - dcm2eul: Rotation matrix (intrinsic) -> Euler angles (multiplication-order)

    - rot2quat: Rotation matrix (extrinsic) -> quaternion
    - dcm2quat: Rotation matrix (intrinsic) -> quaternion
    - quat2rot: Quaternion -> rotation matrix (extrinsic)
    - quat2dcm: Quaternion -> rotation matrix (intrinsic)

    - rotmul: Multiply rotation matrices together
    - rotate: Rotate R3 vector(s) by one or more matrices

    - quatmul: Multiply quaternions together
    - quat_rotate: Rotate R3 vector(s) by one or more quaternions

    - wxyz2rot: (angle, axis) representation -> rotation matrix
    - axang2rot: axis-angle representation -> rotation matrix
    - rot2wxyz: rotation matrix -> (angle, axis) representation
    - rot2axang: rotation matrix -> axis-angle representation

    - wxyz2quat: (angle, axis) representation -> quaternion
    - axang2quat: axis-angle representation -> quaternion
    - quat2wxyz: quaternion -> (angle, axis) representation
    - quat2axang: quaternion -> axis-angle representation

    - convert_angle: deg <-> rad conversion
    - check_orthonormality: checks that determinant is +1

    - e_rot: elementary rotation matrix (extrinsic)
    - e_dcm: elementary direction cosine matrix (intrinsic)

    - angle_between: angle between two vectors
    - a2b_rot: make rotation matrix that rotates vector a to vector b

    - symbolic_e_dcm: Make symbolic (sympy) elementary DCM
    - symbolic_e_rot: Make symbolic (sympy) elementary rotation matrix
    - symbolic_rot: Make symbolic (sympy) rotation matrix
    - symbolic_dcm: Make symbolic (sympy) DCM

Aliases:
    All functions work with stacks of transformations, so "angles"
    is natural in some contexts,

        - convert_angles = convert_angle
        - angles2rot -> angle2rot
        - angles2dcm -> angle2dcm
        - rot2angles -> rot2angle
        - dcm2angles -> dcm2angle

    The Matlab Robotics System Toolbox uses "rotm" for
    "rotation matrix" (rot)

        - eul2rotm -> eul2rot
        - angle2rotm -> angle2rot
        - angles2rotm -> angles2rot

        - rotm2eul -> rot2eul
        - rotm2angle -> rot2angle
        - rotm2angles -> rot2angles

        - rotm2quat -> rot2quat
        - quat2rotm -> quat2rot

        - a2b_rotm -> a2b_rot

        - axang2rotm -> axang2rot
        - rotm2axang -> rot2axang
        - wxyz2rotm -> wxyz2rot
        - rotm2wxyz -> rot2wxyz

This module is completely orthogonal to Viscid, so that it can be
ripped out and used more generally. Please note that Viscid is MIT
licensed, which requires attribution.

The MIT License (MIT)
Copyright (c) 2018 Kristofor Maynard

"""

# pylint: disable = too-many-lines, bad-whitespace, invalid-slice-index

from __future__ import print_function, division
import os
import sys

import numpy as np


__all__ = [# Euler angles <-> rotation matrices
           'angle2rot', 'eul2rot', 'angle2dcm', 'eul2dcm',
           'rot2angles', 'rot2eul', 'dcm2angles', 'dcm2eul',
           # rotation matrices <-> quaternions
           'rot2quat', 'dcm2quat', 'quat2rot', 'quat2dcm',
           # composing rotations / quaternions and applying them in R3
           'rotmul', 'rotate', 'quatmul', 'quat_rotate',
           # axis-angle representation
           'wxyz2rot', 'axang2rot', 'rot2wxyz', 'rot2axang',
           'wxyz2quat', 'axang2quat', 'quat2wxyz', 'quat2axang',
           # misc.
           'convert_angle', 'check_orthonormality',
           'e_rot', 'e_dcm',
           'angle_between', 'a2b_rot',
           # sympy symbolic functions
           'symbolic_e_dcm', 'symbolic_e_rot', 'symbolic_rot', 'symbolic_dcm',
           # aliases
           'convert_angles',
           'angles2rot', 'angles2dcm', 'rot2angles', 'dcm2angles',
           'eul2rotm', 'angle2rotm', 'angles2rotm',
           'rotm2eul', 'rotm2angle', 'rotm2angles',
           'rotm2quat', 'quat2rotm',
           'a2b_rotm', 'axang2rotm', 'rotm2axang', 'wxyz2rotm', 'rotm2wxyz'
           ]


def convert_angle(angle, from_unit, to_unit):
    """convert angle(s) from rad/deg to rad/deg

    Args:
        angle (float, sequence): angle in from_unit
        from_unit (str): unit of angle, one of ('rad', 'deg')
        to_unit (str): unit of result, one of ('rad', 'deg')

    Returns:
        float or ndarray: angle converted to to_unit
    """
    from_unit = from_unit.strip().lower()
    to_unit = to_unit.strip().lower()

    if from_unit.startswith('rad') and to_unit.startswith('rad'):
        ret_angle = angle
    elif from_unit.startswith('deg') and to_unit.startswith('deg'):
        ret_angle = angle
    elif from_unit.startswith('deg') and to_unit.startswith('rad'):
        ret_angle = np.deg2rad(angle)
    elif from_unit.startswith('rad') and to_unit.startswith('deg'):
        ret_angle = np.rad2deg(angle)
    else:
        raise ValueError("Bad angle units '{0}' / '{1}'"
                         "".format(from_unit, to_unit))
    return ret_angle

def check_orthonormality(mat, bad_matrix='warn'):
    """Check that a matrix or stack of matrices is orthonormal

    Args:
        mat (ndarray): A matrix with shape [Ndim, Ndim] or a stack of
            matrices with shape [Nmatrices, Ndim, Ndim]
        bad_matrix (str): What to do if ``numpy.det(R) != 1.0`` - can
            be one of ('ignore', 'warn', 'raise')
    """
    is_valid = None
    bad_matrix = 'ignore' if not bad_matrix else bad_matrix
    bad_matrix = bad_matrix.strip().lower()

    if bad_matrix != 'ignore':
        mat = np.asarray(mat)
        if len(mat.shape) == 2:
            mat = np.reshape(mat, [1] + list(mat.shape))

        is_valid = np.allclose(np.linalg.det(mat), 1.0, atol=8e-16, rtol=4e-16)
        if not is_valid:
            msg = "numpy.linalg.det(mat) != 1.0"
            if bad_matrix == 'raise':
                raise ValueError(msg)
            elif bad_matrix == 'warn':
                print(msg, file=sys.stderr)
    return is_valid

def e_rot(theta, axis='z', unit='rad'):
    """Make elementary rotation matrices (extrinsic) of theta around axis

    Example:

        >>> e_rot(numpy.pi / 2, 'z') @ [1, 0, 0] = [0, 1, 0]

    Args:
        theta (float, sequence): angle or angles
        axis (int, str): one of (0, 'x', 1, 'y', 2, 'z')
        unit (str): unit of theta, one of ('deg', 'rad')

    Returns:
        ndarray: rotation matrix with shape [ndim, ndim] or
            matrices with shape [Nmatrices, Ndim, Ndim]

    Raises:
        ValueError: invalid axis
    """
    theta = np.asarray(theta, dtype=np.double)
    single_val = theta.shape == ()
    theta = theta.reshape(-1)

    theta_rad = convert_angle(theta, unit, 'rad')

    try:
        axis = axis.lower().strip()
    except AttributeError:
        pass

    rotations = np.zeros([len(theta_rad), 3, 3], dtype=np.double)

    sinT = np.sin(theta_rad)
    cosT = np.cos(theta_rad)

    if axis in ('x', 0):
        rotations[:, 0, 0] =   1.0
        rotations[:, 1, 1] =   cosT
        rotations[:, 1, 2] = - sinT
        rotations[:, 2, 1] =   sinT
        rotations[:, 2, 2] =   cosT
    elif axis in ('y', 1):
        rotations[:, 0, 0] =   cosT
        rotations[:, 0, 2] =   sinT
        rotations[:, 1, 1] =   1.0
        rotations[:, 2, 0] = - sinT
        rotations[:, 2, 2] =   cosT
    elif axis in ('z', 2):
        rotations[:, 0, 0] =   cosT
        rotations[:, 0, 1] = - sinT
        rotations[:, 1, 0] =   sinT
        rotations[:, 1, 1] =   cosT
        rotations[:, 2, 2] =   1.0
    else:
        raise ValueError("invalid axis: '{0}'".format(axis))

    if single_val:
        rotations = rotations[0, :, :]
    return rotations

def e_dcm(theta, axis='z', unit='rad'):
    """Make elementary rotation matrices (intrinsic) of theta around axis

    Example:

        >>> e_dcm(numpy.pi / 2, 'z') @ [1, 0, 0] = [0, -1, 0]

    Args:
        theta (float, sequence): angle or angles
        axis (int, str): one of (0, 'x', 1, 'y', 2, 'z')
        unit (str): unit of theta, one of ('deg', 'rad')

    Returns:
        ndarray: rotation matrix with shape [ndim, ndim] or
            matrices with shape [Nmatrices, Ndim, Ndim]

    Raises:
        ValueError: invalid axis
    """
    return e_rot(-np.asarray(theta, dtype=np.double), axis=axis, unit=unit)

def angle2rot(angles, axes='zyx', unit='rad'):
    """Euler angles (transform-order) -> rotation matrix (extrinsic)

    Rotations are applied in transform-order, which means the first
    axis given is the first transform applied. In other words, the
    matrix multiply is in reverse order, i.e.::

        >>> R = (R(angles[..., 2], axis[..., 2]) @
        >>>      R(angles[..., 1], axis[..., 1]) @
        >>>      R(angles[..., 0], axis[..., 0]))

    Example:

        >>> angle2rot([numpy.pi / 2, 0, 0], 'zyx') @ [1, 0, 0] = [0, 1, 0]

    Args:
        angles (sequence): Euler angles in transform-order; can
            have shape [Ndim] or [Nmatrices, Ndim] to make stacked
            transform matrices
        axes (sequence, str): rotation axes in transform-order
        unit (str): unit of angles, one of ('deg', 'rad')

    Returns:
        ndarray: rotation matrix with shape [Ndim, Ndim] or
            [Nmatrices, Ndim, Ndim] depending on the shape of ``angles``
    """
    angles = np.asarray(angles, dtype=np.double)
    single_val = len(angles.shape) == 1
    angles = np.atleast_2d(angles)

    angles_rad = convert_angle(angles, unit, 'rad')

    if angles.shape[-1] != 3:
        raise ValueError("Must have 3 Euler angles")
    if len(axes) != 3:
        raise ValueError("Must have one axis for each Euler angle")

    R0 = e_rot(angles_rad[:, 0], axes[0], unit='rad')
    R1 = e_rot(angles_rad[:, 1], axes[1], unit='rad')
    R2 = e_rot(angles_rad[:, 2], axes[2], unit='rad')

    try:
        # raise AttributeError  # to test einsum timing
        R = np.matmul(R2, np.matmul(R1, R0))
    except AttributeError:
        # fallback to einsum to support numpy 1.6 - 1.9
        matmul_spec = 'mik,mkj->mij'
        R = np.einsum(matmul_spec, R2, np.einsum(matmul_spec, R1, R0))

    if single_val:
        R = R[0, :, :]
    return R

def eul2rot(angles, axes='zyx', unit='rad'):
    """Euler angles (multiplication-order) -> rotation matrix (extrinsic)

    Rotations are applied in multiplication-order, which means the last
    axis given is the first transform applied. In other words::

        >>> R = (R(angles[..., 0], axis[..., 0]) @
        >>>      R(angles[..., 1], axis[..., 1]) @
        >>>      R(angles[..., 2], axis[..., 2]))

    This function is equivalent (up to a few machine epsilon) to
    Matlab's ``eul2rotm``.

    Example:

        >>> eul2rot([numpy.pi / 2, 0, 0], 'zyx') @ [1, 0, 0] = [0, 1, 0]

    Args:
        angles (sequence): Euler angles in multiplication-order; can
            have shape [Ndim] or [Nmatrices, Ndim] to make stacked
            transform matrices
        axes (sequence, str): rotation axes in multiplication-order
        unit (str): unit of angles, one of ('deg', 'rad')

    Returns:
        ndarray: rotation matrix with shape [Ndim, Ndim] or
            [Nmatrices, Ndim, Ndim] depending on the shape of ``angles``
    """
    angles = np.asarray(angles, dtype=np.double)
    single_val = len(angles.shape) == 1
    angles = np.atleast_2d(angles)
    R = angle2rot(angles[:, ::-1], axes=axes[::-1], unit=unit)
    if single_val:
        R = R[0, :]
    return R

def angle2dcm(angles, axes='zyx', unit='rad'):
    """Euler angles (transform-order) -> rotation matrix (intrinsic)

    Rotations are applied in transform-order, which means the first
    axis given is the first transform applied. In other words, the
    matrix multiply is in reverse order, i.e.::

        >>> R = (R(angles[..., 2], axis[..., 2]) @
        >>>      R(angles[..., 1], axis[..., 1]) @
        >>>      R(angles[..., 0], axis[..., 0]))

    This function is equivalent (up to a few machine epsilon) to
    Matlab's ``angle2dcm``.

    Example:

        >>> angle2dcm([numpy.pi / 2, 0, 0], 'zyx') @ [1, 0, 0] = [0, -1, 0]

    Args:
        angles (sequence): Euler angles in transform-order; can
            have shape [Ndim] or [Nmatrices, Ndim] to make stacked
            transform matrices
        axes (sequence, str): rotation axes in transform-order
        unit (str): unit of angles, one of ('deg', 'rad')

    Returns:
        ndarray: rotation matrix with shape [Ndim, Ndim] or
            [Nmatrices, Ndim, Ndim] depending on the shape of ``angles``
    """
    return angle2rot(-1 * np.asarray(angles, dtype=np.double), axes=axes, unit=unit)

def eul2dcm(angles, axes='zyx', unit='rad'):
    """Euler angles (multiplication-order) -> rotation matrix (intrinsic)

    Rotations are applied in multiplication-order, which means the last
    axis given is the first transform applied. In other words::

        >>> R = (R(angles[..., 0], axis[..., 0]) @
        >>>      R(angles[..., 1], axis[..., 1]) @
        >>>      R(angles[..., 2], axis[..., 2]))

    Example:

        >>> eul2dcm([numpy.pi / 2, 0, 0], 'zyx') @ [1, 0, 0] = [0, -1, 0]

    Args:
        angles (sequence): Euler angles in multiplication-order; can
            have shape [Ndim] or [Nmatrices, Ndim] to make stacked
            transform matrices
        axes (sequence, str): rotation axes in multiplication-order
        unit (str): unit of angles, one of ('deg', 'rad')

    Returns:
        ndarray: rotation matrix with shape [Ndim, Ndim] or
            [Nmatrices, Ndim, Ndim] depending on the shape of ``angles``
    """
    angles = np.asarray(angles, dtype=np.double)
    single_val = len(angles.shape) == 1
    angles = np.atleast_2d(angles)
    R = angle2dcm(angles[:, ::-1], axes=axes[::-1], unit=unit)
    if single_val:
        R = R[0, :]
    return R

def rot2angle(R, axes='zyx', unit='rad', bad_matrix='warn'):
    """Rotation matrix (extrinsic) -> Euler angles (transform-order)

    Args:
        R (ndarray): A rotation matrix with shape [Ndim, Ndim] or a
            stack of matrices with shape [Nmatrices, Ndim, Ndim]
        axes (sequence, str): rotation axes in transform-order
        unit (str): Unit of angles, one of ('deg', 'rad')
        bad_matrix (str): What to do if ``numpy.det(R) != 1.0`` - can
            be one of ('ignore', 'warn', 'raise')

    Returns:
        ndarray: Euler angles with shape [Ndim] or [Nmatrices, Ndim]
            depending on the input. ``angles[:, i]`` always corresponds
            to ``axes[i]``.

    See Also:
        * :py:func:`angle2rot` for more details about axes order.
    """
    R = np.asarray(R, dtype=np.double)
    if len(R.shape) == 2:
        R = np.reshape(R, [1] + list(R.shape))
        single_val = True
    else:
        single_val = False

    if R.shape[1:] not in [(3, 3), (4, 4)]:
        raise ValueError("Rotation matrices must be 3x3 or 4x4")
    if len(axes) != 3:
        raise ValueError("Must have one axis for each Euler angle")
    axes = axes.strip().lower()

    check_orthonormality(R, bad_matrix=bad_matrix)

    singular_threshold = 1e-10

    if axes in ('zyx', 'yxz', 'xzy', 'xyz', 'yzx', 'zxy'):
        if axes == 'zyx':
            s1 = R[..., 0, 2]
            c0c1, s0c1 = R[..., 0, 0], - R[..., 0, 1]
            s2c1, c2c1 = - R[..., 1, 2], R[..., 2, 2]
            s0p2, c0p2 = R[..., 2, 1], R[..., 1, 1]
        elif axes == 'yxz':
            s1 = R[..., 2, 1]
            c0c1, s0c1 = R[..., 2, 2], - R[..., 2, 0]
            s2c1, c2c1 = - R[..., 0, 1], R[..., 1, 1]
            s0p2, c0p2 = R[..., 1, 0], R[..., 0, 0]
        elif axes == 'xzy':
            s1 = R[..., 1, 0]
            c0c1, s0c1 = R[..., 1, 1], - R[..., 1, 2]
            s2c1, c2c1 = - R[..., 2, 0], R[..., 0, 0]
            s0p2, c0p2 = R[..., 0, 2], R[..., 2, 2]
        elif axes == 'xyz':
            s1 = - R[..., 2, 0]
            c0c1, s0c1 = R[..., 2, 2], R[..., 2, 1]
            s2c1, c2c1 = R[..., 1, 0], R[..., 0, 0]
            s0p2, c0p2 = - R[..., 0, 1], R[..., 1, 1]
        elif axes == 'yzx':
            s1 = - R[..., 0, 1]
            c0c1, s0c1 = R[..., 0, 0], R[..., 0, 2]
            s2c1, c2c1 = R[..., 2, 1], R[..., 1, 1]
            s0p2, c0p2 = - R[..., 1, 2], R[..., 2, 2]
        elif axes == 'zxy':
            s1 = - R[..., 1, 2]
            c0c1, s0c1 = R[..., 1, 1], R[..., 1, 0]
            s2c1, c2c1 = R[..., 0, 2], R[..., 2, 2]
            s0p2, c0p2 = - R[..., 2, 0], R[..., 0, 0]

        c1 = np.sqrt(c0c1**2 + s0c1**2)
        for arr in (s1, c1, c0c1, s0c1, s2c1, c2c1, s0p2, c0p2):
            np.clip(arr, -1.0, 1.0, arr)
        is_singular = np.abs(c1) < singular_threshold

        angle0 = np.where(is_singular,
                          0.0,
                          np.arctan2(s0c1 / c1, c0c1 / c1)
                          )
        angle1 = np.arctan2(s1, c1)
        angle2 = np.where(is_singular,
                          np.arctan2(s0p2, c0p2),
                          np.arctan2(s2c1 / c1, c2c1 / c1)
                          )
    elif axes in ('zyz', 'zxz', 'yzy', 'yxy', 'xzx', 'xyx'):
        if axes == 'zyz':
            c1 = R[..., 2, 2]
            s0s1, c0s1 = R[..., 2, 1], - R[..., 2, 0]
            s1s2, s1c2 = R[..., 1, 2], R[..., 0, 2]
            s0p2, c0p2 = - R[..., 0, 1], R[..., 1, 1]
        elif axes == 'zxz':
            c1 = R[..., 2, 2]
            s0s1, c0s1 = R[..., 2, 0], R[..., 2, 1]
            s1s2, s1c2 = R[..., 0, 2], - R[..., 1, 2]
            s0p2, c0p2 = R[..., 1, 0], R[..., 0, 0]
        elif axes == 'yzy':
            c1 = R[..., 1, 1]
            s0s1, c0s1 = R[..., 1, 2], R[..., 1, 0]
            s1s2, s1c2 = R[..., 2, 1], - R[..., 0, 1]
            s0p2, c0p2 = R[..., 0, 2], R[..., 2, 2]
        elif axes == 'yxy':
            c1 = R[..., 1, 1]
            s0s1, c0s1 = R[..., 1, 0], - R[..., 1, 2]
            s1s2, s1c2 = R[..., 0, 1], R[..., 2, 1]
            s0p2, c0p2 = - R[..., 2, 0], R[..., 0, 0]
        elif axes == 'xzx':
            c1 = R[..., 0, 0]
            s0s1, c0s1 = R[..., 0, 2], - R[..., 0, 1]
            s1s2, s1c2 = R[..., 2, 0], R[..., 1, 0]
            s0p2, c0p2 = - R[..., 1, 2], R[..., 2, 2]
        elif axes == 'xyx':
            c1 = R[..., 0, 0]
            s0s1, c0s1 = R[..., 0, 1], R[..., 0, 2]
            s1s2, s1c2 = R[..., 1, 0], - R[..., 2, 0]
            s0p2, c0p2 = R[..., 2, 1], R[..., 1, 1]

        s1 = np.sqrt(s0s1**2 + c0s1**2)
        for arr in (s1, c1, s0s1, c0s1, s1s2, s1c2, s0p2, c0p2):
            np.clip(arr, -1.0, 1.0, arr)
        is_singular = np.abs(s1) < singular_threshold

        angle0 = np.where(is_singular,
                          0.0,
                          np.arctan2(s0s1, c0s1)
                          )
        angle1 = np.arctan2(s1, c1)
        angle2 = np.where(is_singular,
                          np.arctan2(s0p2, c0p2),
                          np.arctan2(s1s2, s1c2)
                          )
    else:
        raise ValueError("rot2eul not implemented for order '{0}'".format(axes))

    angles = np.array([angle0, angle1, angle2]).T
    if single_val:
        angles = angles[0, :]
    return convert_angle(angles, 'rad', unit)

def rot2eul(R, axes='zyx', unit='rad', bad_matrix='warn'):
    """Rotation matrix (extrinsic) -> Euler angles (multiplication-order)

    Args:
        R (ndarray): A rotation matrix with shape [Ndim, Ndim] or a
            stack of matrices with shape [Nmatrices, Ndim, Ndim]
        axes (sequence, str): rotation axes in multiplication-order
        unit (str): Unit of angles, one of ('deg', 'rad')
        bad_matrix (str): What to do if ``numpy.det(R) != 1.0`` - can
            be one of ('ignore', 'warn', 'raise')

    Returns:
        ndarray: Euler angles with shape [Ndim] or [Nmatrices, Ndim]
            depending on the input. ``angles[:, i]`` always corresponds
            to ``axes[i]``.

    See Also:
        * :py:func:`rot2eul` for more details about axes order.
    """
    angles = rot2angle(R, axes=axes[::-1], unit=unit, bad_matrix=bad_matrix)
    angles = angles[..., ::-1]
    return angles

def dcm2angle(R, axes='zyx', unit='rad', bad_matrix='warn'):
    """Rotation matrix (intrinsic) -> Euler angles (transform-order)

    Args:
        R (ndarray): A rotation matrix with shape [Ndim, Ndim] or a
            stack of matrices with shape [Nmatrices, Ndim, Ndim]
        axes (sequence, str): rotation axes in transform-order
        unit (str): Unit of angles, one of ('deg', 'rad')
        bad_matrix (str): What to do if ``numpy.det(R) != 1.0`` - can
            be one of ('ignore', 'warn', 'raise')

    Returns:
        ndarray: Euler angles with shape [Ndim] or [Nmatrices, Ndim]
            depending on the input. ``angles[:, i]`` always corresponds
            to ``axes[i]``.

    See Also:
        * :py:func:`angle2dcm` for more details about axes order.
    """
    R = np.asarray(R, dtype=np.double)

    if len(R.shape) == 2:
        R = np.reshape(R, [1] + list(R.shape))
        single_val = True
    else:
        single_val = False

    angles = rot2angle(np.swapaxes(R, -2, -1), axes=axes[::-1], unit=unit,
                       bad_matrix=bad_matrix)[:, ::-1]

    if single_val:
        angles = angles[0, :]

    return angles

def dcm2eul(R, axes='zyx', unit='rad', bad_matrix='warn'):
    """Rotation matrix (intrinsic) -> Euler angles (multiplication-order)

    Args:
        R (ndarray): A rotation matrix with shape [Ndim, Ndim] or a
            stack of matrices with shape [Nmatrices, Ndim, Ndim]
        axes (sequence, str): rotation axes in multiplication-order
        unit (str): Unit of angles, one of ('deg', 'rad')
        bad_matrix (str): What to do if ``numpy.det(R) != 1.0`` - can
            be one of ('ignore', 'warn', 'raise')

    Returns:
        ndarray: Euler angles with shape [Ndim] or [Nmatrices, Ndim]
            depending on the input. ``angles[:, i]`` always corresponds
            to ``axes[i]``.

    See Also:
        * :py:func:`eul2dcm` for more details about axes order.
    """
    angles = dcm2angle(R, axes=axes[::-1], unit=unit, bad_matrix=bad_matrix)
    angles = angles[..., ::-1]
    return angles

def rot2quat(R):
    """Rotation matrix (extrinsic) -> quaternion

    If `this quaternion library <https://github.com/moble/quaternion>`_
    is imported, then the results are presented with dtype quaternion,
    otherwise, as dtype numpy.double (array-of-struct,
    [scalar, vec0, vec1, vec2]).

    Args:
        R (ndarray): A rotation matrix with shape [Ndim, Ndim] or a
            stack of matrices with shape [Nmatrices, Ndim, Ndim]

    Returns:
        ndarray: quaternions with dtype quaternion with shape
            (Nmatrices,) or (); or with dtype numpy.double and shape
            (Nmatrices, 4) or (4)
    """
    R = np.asarray(R, dtype=np.double)
    if len(R.shape) == 2:
        R = np.reshape(R, [1] + list(R.shape))
        single_val = True
    else:
        single_val = False

    if R.shape[1:] not in [(3, 3), (4, 4)]:
        raise ValueError("Rotation matrices must be 3x3 or 4x4")

    K3 = np.zeros([len(R), 4, 4], dtype=np.double)
    K3[:, 0, 0] = (R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2]) / 3.0
    K3[:, 0, 1] = (R[:, 1, 0] + R[:, 0, 1]) / 3.0
    K3[:, 0, 2] = (R[:, 2, 0] + R[:, 0, 2]) / 3.0
    K3[:, 0, 3] = (R[:, 1, 2] - R[:, 2, 1]) / 3.0
    K3[:, 1, 0] = K3[:, 0, 1]
    K3[:, 1, 1] = (R[:, 1, 1] - R[:, 0, 0] - R[:, 2, 2]) / 3.0
    K3[:, 1, 2] = (R[:, 2, 1] + R[:, 1, 2]) / 3.0
    K3[:, 1, 3] = (R[:, 2, 0] - R[:, 0, 2]) / 3.0
    K3[:, 2, 0] = K3[:, 0, 2]
    K3[:, 2, 1] = K3[:, 1, 2]
    K3[:, 2, 2] = (R[:, 2, 2] - R[:, 0, 0] - R[:, 1, 1]) / 3.0
    K3[:, 2, 3] = (R[:, 0, 1] - R[:, 1, 0]) / 3.0
    K3[:, 3, 0] = K3[:, 0, 3]
    K3[:, 3, 1] = K3[:, 1, 3]
    K3[:, 3, 2] = K3[:, 2, 3]
    K3[:, 3, 3] = (R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]) / 3.0

    eigvals, eigvecs = np.linalg.eigh(K3)
    idx_evecs = np.argmax(eigvals, axis=1)
    selected_evecs = eigvecs[np.arange(len(eigvecs)), :, idx_evecs].real
    q = np.empty([len(R), 4], dtype=np.double)
    q[:, 0] = selected_evecs[:, -1]
    q[:, 1:] = - selected_evecs[:, :-1]
    del eigvecs, selected_evecs

    q /= np.linalg.norm(q, axis=-1)[..., np.newaxis]

    if hasattr(np, "quaternion"):
        q = q.view(dtype=np.quaternion)[..., 0]
    if single_val:
        q = q[0]
    return q

def dcm2quat(R):
    """Rotation matrix (intrinsic) -> quaternion

    If `this quaternion library <https://github.com/moble/quaternion>`_
    is imported, then the results are presented with dtype quaternion,
    otherwise, as dtype numpy.double (array-of-struct,
    [scalar, vec0, vec1, vec2]).

    Args:
        R (ndarray): A rotation matrix with shape [Ndim, Ndim] or a
            stack of matrices with shape [Nmatrices, Ndim, Ndim]

    Returns:
        ndarray: quaternions with dtype quaternion and shape
            (Nmatrices,) or (); or with dtype numpy.double and shape
            (Nmatrices, 4) or (4)
    """
    return rot2quat(np.swapaxes(R, -2, -1))

def quat2rot(q):
    """Quaternion -> rotation matrix (extrinsic)

    Args:
        q (ndarray): quaternions with dtype quaternion and shape
            (Nmatrices,) or (); or with dtype numpy.double and shape
            (Nmatrices, 4) or (4)

    Returns:
        ndarray: orthonormal rotation matrix with shape (3, 3)
            or (Nmatrices, 3, 3)
    """
    q = np.asarray(q)
    if hasattr(np, 'quaternion') and q.dtype == 'quaternion':
        single_val = q.shape == ()
        q = q.reshape(-1)
    else:
        single_val = q.shape == (4,)
    q = q.view(dtype=np.double).reshape(-1, 4)

    q_norm = np.linalg.norm(q, axis=-1)
    do_renorm = ~np.bitwise_or(np.isclose(q_norm, 0.0), np.isclose(q_norm, 1.0))
    q[do_renorm, :] /= q_norm[do_renorm].reshape(-1, 1)
    del q_norm, do_renorm

    R = np.empty([len(q), 3, 3], dtype=np.double)

    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y**2 + z**2)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x**2 + z**2)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x**2 + y**2)

    if single_val:
        R = R[0, :, :]
    return R

def quat2dcm(q):
    """Quaternion -> rotation matrix (intrinsic)

    Args:
        q (ndarray): quaternions with dtype quaternion and shape
            (Nmatrices,) or (); or with dtype numpy.double and shape
            (Nmatrices, 4) or (4)

    Returns:
        ndarray: rotation matrix with shape [Ndim, Ndim] or
            [Nmatrices, Ndim, Ndim] depending on the shape of ``q``
    """
    return np.swapaxes(quat2rot(q), -2, -1)

def rotmul(*rmats):
    """Multiply rotation matrices together

    Args:
        *rmats (ndarrays): rotation matrices with shape (3, 3) or
            (nmats, 3, 3)

    Returns:
        ndarray: rotation matrix with shape (3, 3) or (nmats, 3, 3)
    """
    single_val = True
    R = np.eye(3, dtype=np.double).reshape(1, 3, 3)

    for rmat in rmats:
        rmat = np.asarray(rmat, dtype=np.double)
        if len(rmat.shape) > 2:
            single_val = False
        rmat = rmat.reshape(-1, 3, 3)

        R = np.einsum('mij,mjk->mik', R, rmat)

    if single_val:
        R = R[0, :, :]
    return R

def rotate(vec, *rmats):
    """Rotate R3 vector(s) by one or more matrices

    Args:
        vec (ndarray): R3 vectors with shape (3,) or (nvecs, 3)
        *rmats (ndarrays): rotation matrices with shape (3, 3) or
            (nmats, 3, 3)

    Returns:
        ndarray: rotated vectors with shape (3,) or (nvecs, 3)
    """
    R = rotmul(*rmats)
    vec = np.asarray(vec, dtype=np.double)
    single_val = len(vec.shape) == 1 and len(R.shape) == 2
    vec = vec.reshape(-1, 3)
    R = R.reshape(-1, 3, 3)

    r_vec = np.einsum('mij,mj->mi', R, vec)

    if single_val:
        r_vec = r_vec[0, :]
    return r_vec

def quatmul(*quats):
    """Multiply quaternions together

    Args:
        *quats (ndarrays): quaternions with dtype quaternion and shape
            () or (nquat,); or with dtype np.double and shape (4,) or
            (nquat, 4)

    Returns:
        ndarray: with dtype quaternion and shape () or (nquat,); or
            with dtype np.double and shape (4,) or (nquat, 4)
    """
    single_val = True

    if hasattr(np, 'quaternion'):
        q0 = np.array([1, 0, 0, 0], dtype=np.double).view(dtype='quaternion')

        for q in quats:
            q = np.asarray(q)
            if q.dtype == 'quaternion':
                if q.shape != ():
                    single_val = False
            else:
                if len(q.shape) > 1:
                    single_val = False
                q = q.view(dtype='quaternion')
            q = q.reshape(-1)
            q0 = q0 * q
    else:
        q0 = np.array([1, 0, 0, 0], dtype=np.double).reshape(1, 4)
        r = np.zeros([1, 4], dtype=np.double)

        for q in quats:
            q = np.asarray(q, dtype=np.double)
            if len(q.shape) > 1:
                single_val = False
            q = q.reshape(-1, 4)
            a0, a1, a2, a3 = q0[:, 0], q0[:, 1], q0[:, 2], q0[:, 3]
            b0, b1, b2, b3 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            if len(q) > len(q0):
                q0 = (q0.reshape(1, -1)
                      * np.array([1] + [0] * (len(q) - len(q0))).reshape(-1, 1))
            if len(q) > len(r):
                r = (r.reshape(1, -1)
                      * np.array([1] + [0] * (len(q) - len(r))).reshape(-1, 1))

            r[:, 0] = a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3
            r[:, 1] = a1 * b0 + a0 * b1 - a3 * b2 + a2 * b3
            r[:, 2] = a2 * b0 + a3 * b1 + a0 * b2 - a1 * b3
            r[:, 3] = a3 * b0 - a2 * b1 + a1 * b2 + a0 * b3
            q0[:, :] = r[:, :]

    if single_val:
        q0 = q0[0]
    return q0

def quat_rotate(vec, *quats):
    """Rotate R3 vector(s) by one or more quaternions

    Args:
        vec (ndarray): R3 vectors with shape (3,) or (nvecs, 3)
        *quats (ndarrays): quaternions with dtype quaternion and shape
            () or (nquat,); or with dtype np.double and shape (4,) or
            (nquat, 4)

    Returns:
        ndarray: rotated vectors with shape (3,) or (nvecs, 3)
    """
    q = quatmul(*quats)
    vec = np.asarray(vec, dtype=np.double)

    # single_val = len(vec.shape) == 1 #and
    if hasattr(np, 'quaternion') and q.dtype == 'quaternion':
        single_val = len(vec.shape) == 1 and q.shape == ()
    else:
        single_val = len(vec.shape) == 1 and q.shape == (4,)

    # turn input R3 vector into quaternion
    vec = vec.reshape(-1, 3)
    p = np.zeros([len(vec), 4], dtype=np.double)
    p[:, 1:] = vec[:, :]
    del vec

    q_inv = np.copy(q).reshape(-1).view(dtype=np.double).reshape(-1, 4)
    q_inv[:, 1:] *= -1
    q_inv /= np.sum(q_inv**2, axis=1)[:, np.newaxis]

    r_quat = quatmul(q, p, q_inv)
    r = np.array(r_quat.view(dtype=np.double).reshape(-1, 4)[:, 1:])

    if single_val:
        r = r[0, :]
    return r

def wxyz2rot(angle, vector, unit='rad'):
    """Make a matrix (matrices) that rotates around vector by angle

    Args:
        angle (sequence): angle(s) of rotation(s) with
            shape (), (1,), or (Nmatrices)
        vector (sequence): 3d vector(s) that is (are) axis of rotation(s)
            with shape (3,) or (Nmatrices, 3)
        unit (str): unit of angle, one of ('deg', 'rad')

    Returns:
        ndarray: orthonormal rotation matrix with shape (3, 3)
            or (Nmatrices, 3, 3)
    """
    angle = np.asarray(angle, dtype=np.double)
    vector = np.asarray(vector, dtype=np.double)

    single_val = len(angle.shape) == 0 and len(vector.shape) == 1

    if len(angle.shape) <= 2:
        angle = angle.reshape(-1, 1, 1)
    vector = np.atleast_2d(vector)

    theta_rad = convert_angle(angle, unit, 'rad')
    k = vector / np.linalg.norm(vector, axis=1).reshape(-1, 1)

    Nmatrices = max(angle.shape[0], vector.shape[0])
    K = np.zeros([Nmatrices, 3, 3], dtype=np.double)

    K[:, 0, 1] = -k[:, 2]
    K[:, 0, 2] =  k[:, 1]
    K[:, 1, 0] =  k[:, 2]
    K[:, 1, 2] = -k[:, 0]
    K[:, 2, 0] = -k[:, 1]
    K[:, 2, 1] =  k[:, 0]

    try:
        Ksq = np.matmul(K, K)
    except AttributeError:
        Ksq = np.einsum('mik,mkj->mij', K, K)

    R = (np.eye(3).reshape(1, 3, 3)
         + np.sin(theta_rad) * K
         + (1 - np.cos(theta_rad)) * Ksq)

    if single_val:
        R = R[0, :, :]

    return R

def axang2rot(axang, unit='rad'):
    """Make a matrix (matrices) that rotates around vector by angle

    Args:
        axang (ndarray): axis(es) / angle(s) of rotation(s) put
            together like {x, y, z, angle}; shape should be (4,), or
            (Nmatrices, 4)
        unit (str): unit of angle, one of ('deg', 'rad')

    Returns:
        ndarray: orthonormal rotation matrix with shape (3, 3)
            or (Nmatrices, 3, 3)
    """
    axang = np.asarray(axang, dtype=np.double)
    angle = axang[..., 3]
    axis = axang[..., :3]
    return wxyz2rot(angle, axis, unit=unit)

def rot2wxyz(R, unit='rad', quick=True, bad_matrix='warn'):
    """Find axis / angle representation of rotation matrix (matrices)

    Args:
        R (ndarray): Rotation matrix with shape (3, 3) determinant +1,
            or matrices with shape (Nmatrices, 3, 3)
        unit (str): unit of angle in results, one of ('deg', 'rad')
        quick (bool): Try a quicker, less well tested method
        bad_matrix (str): What to do if ``numpy.det(R) != 1.0`` - can
            be one of ('ignore', 'warn', 'raise')

    Returns:
        (w, xyz): ``w`` is rotation angle with shape () or (Nmatrices),
            and ``xyz`` are normalized rotation axes with shape (3,)
            or (Nmatrices, 3)
    """
    R = np.asarray(R, dtype=np.double)
    if len(R.shape) == 2:
        R = np.reshape(R, [1] + list(R.shape))
        single_val = True
    else:
        single_val = False

    if R.shape[1:] != (3, 3):
        raise ValueError("Rotation matrices must be 3x3")

    check_orthonormality(R, bad_matrix=bad_matrix)

    Nmatrices = R.shape[0]

    if quick:
        # I'm not sure if I trust this method, although it is probably
        # quicker than calculating eigenvectors
        ux = R[:, 2, 1] - R[:, 1, 2]
        uy = R[:, 0, 2] - R[:, 2, 0]
        uz = R[:, 1, 0] - R[:, 0, 1]
        u = np.array([ux, uy, uz]).T
    else:
        u = np.zeros([Nmatrices, 3], dtype=R.dtype)

    null_mask = np.isclose(np.linalg.norm(u, axis=1), 0.0)

    if any(null_mask):
        # calculate eigen vectors for transforms where the quick
        # method failed (or was never calculated)
        eigval, eigvecs = np.linalg.eigh(R[null_mask])
        idx_evec = np.argmin(np.abs(eigval - 1.0), axis=1)
        # eigenvector for eigenval closest to 1.0
        u[null_mask, :] = eigvecs[np.arange(len(eigvecs)), :, idx_evec].real

    # normalize axis vector
    u = u / np.linalg.norm(u, axis=1).reshape(-1, 1)

    # find a vector `b1` that is perpendicular to rotation axis `u`
    a = np.zeros([Nmatrices, 3], dtype=R.dtype)
    a[:, 0] = 1.0
    au_diff = np.linalg.norm(a - u, axis=1)
    # if au_diff < 0.1 or au_diff > 1.9:
    a[np.bitwise_or(au_diff < 0.1, au_diff > 1.9), :] = [0, 1, 0]
    b1 = np.cross(a, u, axis=1)
    # renormalize to minimize roundoff error
    b1 /= np.linalg.norm(b1, axis=1).reshape(-1, 1)

    # rotate `b1` and then find the angle between result (`b2`) at `b1`
    b2 = np.einsum('mij,mj->mi', R, b1)
    b2 /= np.linalg.norm(b2, axis=1).reshape(-1, 1)
    angleB = np.arccos(np.clip(np.sum(b1 * b2, axis=1), -1.0, 1.0))

    # fix sign of `u` to make the rotation angle right handed, note that
    # the angle will always be positive due to range or np.arccos
    neg_mask = np.sum(np.cross(b1, b2, axis=1) * u, axis=1) < 0.0
    u[neg_mask, :] *= -1

    angleB = convert_angle(angleB, 'rad', unit)

    if single_val:
        angleB = angleB[0]
        u = u[0, :]

    return angleB, u

def rot2axang(R, unit='rad', quick=True, bad_matrix='warn'):
    """Find axis / angle representation of rotation matrix (matrices)

    Args:
        R (ndarray): Rotation matrix with shape (3, 3) determinant +1,
            or matrices with shape (Nmatrices, 3, 3)
        unit (str): unit of angle in results, one of ('deg', 'rad')
        quick (bool): Try a quicker, less well tested method
        bad_matrix (str): What to do if ``numpy.det(R) != 1.0`` - can
            be one of ('ignore', 'warn', 'raise')

    Returns:
        (ndarray): {x, y, z, angle} axis / angle values with shape (4,)
            or (Nmatrices, 4)
    """
    ang, ax = rot2wxyz(R, unit=unit, quick=quick, bad_matrix=bad_matrix)
    if len(ax.shape) == 1:
        axang = np.empty([4], dtype=ax.dtype)
    else:
        axang = np.empty([ax.shape[0], 4], dtype=ax.dtype)
    axang[..., :3] = ax
    axang[..., 3] = ang
    return axang

def wxyz2quat(angle, vector, unit='rad'):
    """(angle, axis) representation -> quaternion

    Args:
        angle (sequence): angle(s) of rotation(s) with
            shape (), (1,), or (Nmatrices)
        vector (sequence): 3d vector(s) that is (are) axis of rotation(s)
            with shape (3,) or (Nmatrices, 3)
        unit (str): unit of angle, one of ('deg', 'rad')

    Returns:
        ndarray: quaternions with dtype quaternion and shape
            (Nmatrices,) or (); or with dtype numpy.double and shape
            (Nmatrices, 4) or (4)
    """
    angle = np.asarray(angle, dtype=np.double)
    vector = np.asarray(vector, dtype=np.double)

    single_val = len(angle.shape) == 0 and len(vector.shape) == 1

    angle = angle.reshape(-1, 1)
    vector = vector.reshape(-1, 3)

    half_theta_rad = convert_angle(angle, unit, 'rad') / 2

    q = np.zeros([len(half_theta_rad), 4], dtype=np.double)

    q[:, 0:1] = np.cos(half_theta_rad)
    q[:, 1:] = (np.sin(half_theta_rad)
                * vector / np.linalg.norm(vector, axis=1)[:, np.newaxis])

    if hasattr(np, 'quaternion'):
        q = q.view(dtype='quaternion').reshape(-1)
    if single_val:
        q = q[0]
    return q

def axang2quat(axang, unit='rad'):
    """axis-angle representation -> quaternion

    Args:
        axang (ndarray): axis(es) / angle(s) of rotation(s) put
            together like {x, y, z, angle}; shape should be (4,), or
            (Nmatrices, 4)
        unit (str): unit of angle, one of ('deg', 'rad')

    Returns:
        ndarray: quaternions with dtype quaternion and shape
            (Nmatrices,) or (); or with dtype numpy.double and shape
            (Nmatrices, 4) or (4)
    """
    axang = np.asarray(axang, dtype=np.double)
    angle = axang[..., 3]
    axis = axang[..., :3]
    return wxyz2quat(angle, axis, unit=unit)

def quat2wxyz(q, unit='rad'):
    """quaternion -> (angle, axis) representation

    Args:
        q (ndarray): quaternions with dtype quaternion and shape
            (Nmatrices,) or (); or with dtype numpy.double and shape
            (Nmatrices, 4) or (4)
        unit (str): unit of angle in results, one of ('deg', 'rad')

    Returns:
        (w, xyz): ``w`` is rotation angle with shape () or (Nmatrices),
            and ``xyz`` are normalized rotation axes with shape (3,)
            or (Nmatrices, 3)
    """
    q = np.asarray(q)
    if hasattr(np, 'quaternion') and q.dtype == 'quaternion':
        single_val = q.shape == ()
        q = q.view(np.double).reshape(-1, 4)
    else:
        single_val = len(q.shape) == 1
        q = q.astype(np.double).reshape(-1, 4)

    wxyz = np.zeros([len(q), 4], dtype=np.double)

    # normalize quats
    q = q / np.linalg.norm(q, axis=1)[-1, np.newaxis]

    vec_norm = np.linalg.norm(q[:, 1:], axis=1)
    wxyz[:, 0] = 2 * np.arctan2(vec_norm, q[:, 0])
    with np.errstate(divide='ignore', invalid='ignore'):
        wxyz[:, 1:] = q[:, 1:] / vec_norm[:, np.newaxis]
        mask = np.isclose(vec_norm, 0.0)
        wxyz[mask, 1:] = [[0, 0, 1]]

    wxyz[:, 0] = convert_angles(wxyz[:, 0], 'rad', unit)
    if single_val:
        wxyz = wxyz[0, :]
    return wxyz[..., 0], wxyz[..., 1:]

def quat2axang(q, unit='rad'):
    """quaternion -> axis-angle representation

    Args:
        q (ndarray): quaternions with dtype quaternion and shape
            (Nmatrices,) or (); or with dtype numpy.double and shape
            (Nmatrices, 4) or (4)
        unit (str): unit of angle in results, one of ('deg', 'rad')

    Returns:
        (ndarray): {x, y, z, angle} axis / angle values with shape (4,)
            or (Nmatrices, 4)
    """
    ang, ax = quat2wxyz(q, unit=unit)
    if len(ax.shape) == 1:
        axang = np.empty([4], dtype=ax.dtype)
    else:
        axang = np.empty([ax.shape[0], 4], dtype=ax.dtype)
    axang[..., :3] = ax
    axang[..., 3] = ang
    return axang

def angle_between(a, b, unit='rad'):
    """Get angle(s) between two (sets of) vectors

    Args:
        a (sequence): first vector, shape [Ndim] or [Nvectors, Ndim]
        b (sequence): second vector, shape [Ndim] or [Nvectors, Ndim]
        unit (str): unit of result, one of ('deg', 'rad')

    Returns:
        float or ndarray: angle(s)
    """
    a = np.asarray(a)
    b = np.asarray(b)
    single_val = len(a.shape) == 1 and len(b.shape) == 1
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)

    a_norm = np.linalg.norm(a, axis=-1)
    b_norm = np.linalg.norm(b, axis=-1)

    normdot = np.sum(a * b, axis=-1) / (a_norm * b_norm)
    angle_rad = np.arccos(normdot)

    if single_val:
        angle_rad = angle_rad[0]
    return convert_angle(angle_rad, 'rad', unit)

def a2b_rot(a, b, roll=0.0, unit='rad', new_x=None):
    """Make a matrix that rotates vector a to vector b

    Args:
        a (ndarray): starting vector with shape [Ndim]
            or [Nvectors, Ndim]
        b (ndarray): destination vector with shape [Ndim]
            or [Nvectors, Ndim]
        roll (float): angle of roll around the b vector
        unit (str): unit of roll, one of ('deg', 'rad')
        new_x (ndarray): If given, then roll is set such that
            the `x` axis (phi = 0) new_x is projected in the plane
            perpendicular to origin-p1

    Returns:
        ndarray: orthonormal rotation matrix with shape [Ndim, Ndim]
            or [Nvectors, Ndim, Ndim]
    """
    a = np.asarray(a, dtype=np.double)
    b = np.asarray(b, dtype=np.double)
    roll = np.asarray(roll, dtype=np.double)

    single_val = len(a.shape) == 1 and len(b.shape) == 1 and len(roll.shape) == 0

    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    roll = np.atleast_2d(roll)

    a /= np.linalg.norm(a, axis=1).reshape(-1, 1)
    b /= np.linalg.norm(b, axis=1).reshape(-1, 1)

    theta_rad = np.arctan2(np.linalg.norm(np.cross(a, b, axis=1), axis=1),
                           np.sum(a * b, axis=1))
    axis = np.cross(a, b, axis=1)
    null_mask = np.isclose(np.linalg.norm(axis, axis=1), 0.0)
    if any(null_mask):
        axis[null_mask, :] = np.cross(a[null_mask, :], [[1, 0, 0]], axis=1)
        null_mask = np.isclose(np.linalg.norm(axis, axis=1), 0.0)
        if any(null_mask):
            axis[null_mask, :] = np.cross(a[null_mask, :], [[0, 1, 0]], axis=1)

    R = wxyz2rot(theta_rad, axis)

    # optionally set roll using a desired orientation for the x-axis
    if new_x is not None:
        assert roll == 0.0
        new_x = np.asarray(new_x, dtype=np.double).reshape(-1, 3)
        new_x = new_x - np.sum(new_x * b, axis=1)[:, np.newaxis] * b
        new_x_norm = np.linalg.norm(new_x, axis=1)[:, np.newaxis]
        roll = np.zeros([len(new_x)], dtype=np.double)
        no_norm = np.isclose(new_x_norm[:, 0], 0.0)
        roll[no_norm] = 0.0
        if any(~no_norm):
            current_x = np.einsum('mij,mj->mi', R, [[1, 0, 0]])
            new_x /= new_x_norm
            current_x /= np.linalg.norm(current_x, axis=1)[:, np.newaxis]
            s_roll = np.linalg.norm(np.cross(current_x, new_x, axis=1), axis=1)
            c_roll = np.sum(current_x * new_x, axis=1)
            roll[~no_norm] = np.arctan2(s_roll, c_roll)
            roll[~no_norm] = convert_angle(roll[~no_norm], 'rad', unit)
            roll[~no_norm] *= np.sign(np.sum(np.cross(current_x, new_x, axis=1)
                                             * b, axis=1))

    if np.any(roll):
        roll_rad = convert_angle(roll, unit, 'rad')
        R_roll = wxyz2rot(roll_rad, b)
        try:
            R = np.matmul(R_roll, R)
        except AttributeError:
            R = np.einsum('mik,mkj->mij', R_roll, R)

    if single_val:
        R = R[0, :, :]
    return R

def symbolic_e_dcm(axis='z', angle=None):
    """Make elementary matrix that rotate axes, not vectors (sympy)"""
    import sympy
    axis = axis.strip().lower()
    if angle is None:
        angle = sympy.Symbol('x')

    if axis == 'x':
        ret = sympy.rot_axis1(angle)
    elif axis == 'y':
        ret = sympy.rot_axis2(angle)
    elif axis == 'z':
        ret = sympy.rot_axis3(angle)
    else:
        raise ValueError("Unrecognized axis: '{0}'".format(axis))
    return ret

def symbolic_e_rot(axis='z', angle=None):
    """Make elementary matrix that rotate vectors, not the axes (sympy)"""
    return symbolic_e_dcm(axis=axis, angle=angle).T

def symbolic_rot(axes='zyx', angles=None):
    """Make a symbolic rotation matrix (sympy)"""
    import sympy
    if angles is None:
        angles = [sympy.Symbol(str(i)) for i, _ in enumerate(axes)]
    R = sympy.eye(3)
    for axis, angle in zip(axes[::-1], angles[::-1]):
        R = R * symbolic_e_rot(axis=axis, angle=angle)
    return R

def symbolic_dcm(axes='zyx', angles=None):
    """Make a symbolic direction cosine matrix (sympy)"""
    import sympy
    if angles is None:
        angles = [sympy.Symbol(str(i)) for i, _ in enumerate(axes)]
    R = sympy.eye(3)
    for axis, angle in zip(axes[::-1], angles[::-1]):
        R = R * symbolic_e_dcm(axis=axis, angle=angle)
    return R

###########
## Aliases
###########

convert_angles = convert_angle

angles2rot = angle2rot
angles2dcm = angle2dcm
rot2angles = rot2angle
dcm2angles = dcm2angle

# The Matlab Robotics System Toolbox uses "rotm" for "rotation matrix" (rot),
# and "axang" for "axis-angle" (wxyz),

eul2rotm = eul2rot
angle2rotm = angle2rot
angles2rotm = angles2rot

rotm2eul = rot2eul
rotm2angle = rot2angle
rotm2angles = rot2angle

rotm2quat = rot2quat
quat2rotm = quat2rot

a2b_rotm = a2b_rot

axang2rotm = axang2rot
rotm2axang = rot2axang

wxyz2rotm = wxyz2rot
rotm2wxyz = rot2wxyz

########
# Tests
########

def _print(*args, **kwargs):
    kwargs['file'] = sys.stderr
    print(*args, **kwargs)

def _check_return_shapes():
    _print("Checking return shapes")
    assert angle_between([1, 0, 0], [0, 1, 0]).shape == ()
    assert angle_between([[1, 0, 0]], [[0, 1, 0]]).shape == (1,)
    assert angle_between([[1, 0, 0], [0, 1, 0]],
                         [[0, 1, 0], [1, 0, 0]]).shape == (2,)

    assert angle2rot([np.pi / 6, np.pi / 3, np.pi / 2]).shape == (3, 3)
    assert angle2rot([[np.pi / 6, np.pi / 3, np.pi / 2]]).shape == (1, 3, 3)
    assert angle2rot([[np.pi / 6, np.pi / 3, np.pi / 2],
                      [np.pi / 2, np.pi / 3, np.pi / 6]]).shape == (2, 3, 3)

    assert angle2dcm([np.pi / 6, np.pi / 3, np.pi / 2]).shape == (3, 3)
    assert angle2dcm([[np.pi / 6, np.pi / 3, np.pi / 2]]).shape == (1, 3, 3)
    assert angle2dcm([[np.pi / 6, np.pi / 3, np.pi / 2],
                      [np.pi / 2, np.pi / 3, np.pi / 6]]).shape == (2, 3, 3)

    assert eul2rot([np.pi / 6, np.pi / 3, np.pi / 2]).shape == (3, 3)
    assert eul2rot([[np.pi / 6, np.pi / 3, np.pi / 2]]).shape == (1, 3, 3)
    assert eul2rot([[np.pi / 6, np.pi / 3, np.pi / 2],
                    [np.pi / 2, np.pi / 3, np.pi / 6]]).shape == (2, 3, 3)

    assert eul2dcm([np.pi / 6, np.pi / 3, np.pi / 2]).shape == (3, 3)
    assert eul2dcm([[np.pi / 6, np.pi / 3, np.pi / 2]]).shape == (1, 3, 3)
    assert eul2dcm([[np.pi / 6, np.pi / 3, np.pi / 2],
                    [np.pi / 2, np.pi / 3, np.pi / 6]]).shape == (2, 3, 3)

    assert rot2angle(np.eye(3)).shape == (3,)
    assert rot2angle([np.eye(3)]).shape == (1, 3)
    assert rot2angle([np.eye(3), np.diag([1, -1, -1])]).shape == (2, 3)

    assert rot2eul(np.eye(3)).shape == (3,)
    assert rot2eul([np.eye(3)]).shape == (1, 3)
    assert rot2eul([np.eye(3), np.diag([1, -1, -1])]).shape == (2, 3)

    assert dcm2angle(np.eye(3)).shape == (3,)
    assert dcm2angle([np.eye(3)]).shape == (1, 3)
    assert dcm2angle([np.eye(3), np.diag([1, -1, -1])]).shape == (2, 3)

    assert dcm2eul(np.eye(3)).shape == (3,)
    assert dcm2eul([np.eye(3)]).shape == (1, 3)
    assert dcm2eul([np.eye(3), np.diag([1, -1, -1])]).shape == (2, 3)

    assert rotm2quat(np.eye(3)).shape == (4,)
    assert rotm2quat([np.eye(3)]).shape == (1, 4)
    assert rotm2quat([np.eye(3), np.eye(3)]).shape == (2, 4)
    assert dcm2quat(np.eye(3)).shape == (4,)
    assert dcm2quat([np.eye(3)]).shape == (1, 4)
    assert dcm2quat([np.eye(3), np.eye(3)]).shape == (2, 4)

    assert quat2rotm([1, 0, 0, 0]).shape == (3, 3)
    assert quat2rotm([[1, 0, 0, 0]]).shape == (1, 3, 3)
    assert quat2rotm([[1, 0, 0, 0], [1, 0, 0, 0]]).shape == (2, 3, 3)
    assert quat2dcm([1, 0, 0, 0]).shape == (3, 3)
    assert quat2dcm([[1, 0, 0, 0]]).shape == (1, 3, 3)
    assert quat2dcm([[1, 0, 0, 0], [2, 0, 0, 0]]).shape == (2, 3, 3)

    assert a2b_rot([1, 0, 0], [0, 1, 0]).shape == (3, 3)
    assert a2b_rot([[1, 0, 0]], [[0, 1, 0]]).shape == (1, 3, 3)
    assert a2b_rot([[1, 0, 0], [0, 1, 0]],
                   [[0, 1, 0], [1, 0, 0]]).shape == (2, 3, 3)

    assert rotmul(np.eye(3)).shape == (3, 3)
    assert rotmul([np.eye(3)]).shape == (1, 3, 3)
    assert rotmul(np.eye(3), [np.eye(3)]).shape == (1, 3, 3)
    assert rotmul(np.eye(3), [np.eye(3), np.eye(3)]).shape == (2, 3, 3)

    assert rotate([1, 0, 0], np.eye(3)).shape == (3,)
    assert rotate([1, 0, 0], [np.eye(3)]).shape == (1, 3)
    assert rotate([1, 0, 0], np.eye(3), [np.eye(3)]).shape == (1, 3)
    assert rotate([1, 0, 0], np.eye(3), [np.eye(3), np.eye(3)]).shape == (2, 3)
    assert rotate([[1, 0, 0], [1, 0, 0]], np.eye(3), np.eye(3)).shape == (2, 3)

    assert quatmul([1, 0, 0, 0], [1, 0, 0, 0]).shape == (4,)
    assert quatmul([1, 0, 0, 0], [[1, 0, 0, 0]]).shape == (1, 4)
    assert quatmul([1, 0, 0, 0], [[1, 0, 0, 0], [1, 0, 0, 0]]).shape == (2, 4)

    assert quat_rotate([1, 0, 0], [1, 0, 0, 0]).shape == (3,)
    assert quat_rotate([[1, 0, 0]], [1, 0, 0, 0]).shape == (1, 3)
    assert quat_rotate([1, 0, 0], [[1, 0, 0, 0]]).shape == (1, 3)
    assert quat_rotate([1, 0, 0], [[1, 0, 0, 0], [1, 0, 0, 0]]).shape == (2, 3)
    assert quat_rotate([[1, 0, 0], [1, 0, 0]], [1, 0, 0, 0]).shape == (2, 3)

    assert wxyz2rot(np.pi / 2, [0, 0, 1]).shape == (3, 3)
    assert wxyz2rot([np.pi / 2], [[0, 0, 1]]).shape == (1, 3, 3)
    assert wxyz2rot([np.pi / 2, np.pi],
                    [[0, 0, 1], [0, 0, -1]]).shape == (2, 3, 3)
    assert axang2rot([0, 0, 1, np.pi / 2]).shape == (3, 3)
    assert axang2rot([[0, 0, 1, np.pi / 2]]).shape == (1, 3, 3)
    assert axang2rot([[0, 0, 1, np.pi / 2], [0, 0, -1, np.pi]]).shape == (2, 3, 3)

    assert rot2wxyz(np.eye(3))[0].shape == ()
    assert rot2wxyz(np.eye(3))[1].shape == (3,)
    assert rot2wxyz([np.eye(3)])[0].shape == (1,)
    assert rot2wxyz([np.eye(3)])[1].shape == (1, 3)
    assert rot2wxyz([np.eye(3), np.eye(3)])[0].shape == (2,)
    assert rot2wxyz([np.eye(3), np.eye(3)])[1].shape == (2, 3)
    assert rot2axang(np.eye(3)).shape == (4,)
    assert rot2axang([np.eye(3)]).shape == (1, 4)
    assert rot2axang([np.eye(3), np.eye(3)]).shape == (2, 4)

    assert wxyz2quat(np.pi / 2, [0, 0, 1]).shape == (4,)
    assert wxyz2quat([np.pi / 2], [[0, 0, 1]]).shape == (1, 4)
    assert wxyz2quat([np.pi / 2, np.pi],
                    [[0, 0, 1], [0, 0, -1]]).shape == (2, 4)
    assert axang2quat([0, 0, 1, np.pi / 2]).shape == (4,)
    assert axang2quat([[0, 0, 1, np.pi / 2]]).shape == (1, 4)
    assert axang2quat([[0, 0, 1, np.pi / 2], [0, 0, -1, np.pi]]).shape == (2, 4)

    assert quat2wxyz([1, 0, 0, 0])[0].shape == ()
    assert quat2wxyz([1, 0, 0, 0])[1].shape == (3,)
    assert quat2wxyz([[1, 0, 0, 0]])[0].shape == (1,)
    assert quat2wxyz([[1, 0, 0, 0]])[1].shape == (1, 3)
    assert quat2wxyz([[1, 0, 0, 0], [1, 0, 0, 0]])[0].shape == (2,)
    assert quat2wxyz([[1, 0, 0, 0], [1, 0, 0, 0]])[1].shape == (2, 3)
    assert quat2axang([1, 0, 0, 0]).shape == (4,)
    assert quat2axang([[1, 0, 0, 0]]).shape == (1, 4)
    assert quat2axang([[1, 0, 0, 0], [1, 0, 0, 0]]).shape == (2, 4)

    return 0

def _check_angle2matrix():
    _print("Checking angle2matrix")

    ret = 0

    assert np.allclose(np.einsum('ij,j->i',
                                 angle2rotm([np.pi / 2, 0, -np.pi / 2], 'zyx'),
                                 [1, 0, 0]),
                       [0, 0, -1])
    assert np.allclose(np.einsum('ij,j->i',
                                 angle2dcm([np.pi / 2, 0, -np.pi / 2], 'zyx'),
                                 [1, 0, 0]),
                       [0, 0, -1])

    assert np.allclose(np.einsum('ij,j->i',
                                 eul2rotm([np.pi / 2, 0, -np.pi / 2], 'zyx'),
                                 [1, 0, 0]),
                       [0, 1, 0])
    assert np.allclose(np.einsum('ij,j->i',
                                 eul2dcm([np.pi / 2, 0, -np.pi / 2], 'zyx'),
                                 [1, 0, 0]),
                       [0, -1, 0])

    for fn, fname in zip((angle2dcm, eul2rotm), ('angle2dcm.csv', 'eul2rotm.csv')):
        _print("Checking angle2dcm")
        try:
            import pandas as pd
            if os.path.isfile(fname):
                df = pd.read_csv(fname)
            else:
                raise OSError
        except OSError:
            _print("'{0}' does not exist, skipping test\n"
                   "    (it can be generated using Viscid/scratch/gen_rotations.m)"
                   "".format(fname))
            return 0
        except ImportError:
            _print("Pandas not installed, skipping test")
            return 0

        unique_order = [s for s in sorted(df['order'].unique())]

        # nrows = int(df.shape[0] // len(unique_order))
        orders = df.loc[:, 'order'].values.astype('U3')
        matlab_angles = df.loc[:, 'ang0':'ang2'].values.reshape(-1, 3).astype('f8')
        matlab_dcms = df.loc[:, 'R00':'R22'].values.reshape(-1, 3, 3).astype('f8')

        for _, order in enumerate(unique_order):
            _print(order, '... ', sep='', end='')
            zyx_slice = np.argwhere(orders == order).reshape(-1)
            mlb_angs = matlab_angles[zyx_slice]
            mlb_mats = matlab_dcms[zyx_slice]
            # mlb_xangs = matlab_xangles[zyx_slice]

            py_mats = fn(mlb_angs, order)

            success = True
            for i in range(mlb_angs.shape[0]):
                if not np.allclose(py_mats[i], mlb_mats[i], atol=4e-12, rtol=4e-12):
                    success = False
                    ret += 1
                    _print("Failure (", i, ')', sep='')
                    _print("    angles:", mlb_angs[i])
                    _print("    matlab dcm:\n", mlb_mats[i], sep='')
                    _print("    python dcm:\n", py_mats[i], sep='')
            if success:
                _print('Success!')
    return ret

def _check_matrix2angle(quick=False):
    ret = 0

    if quick:
        orders = ('zyx', 'zyz')
        arr = np.deg2rad([-1.0, -1e-3, 0, 1e-3, 1.0])
        clusters = np.deg2rad([-360, -270, -180, -135, -90, -60, 0,
                               30, 90, 135, 180, 270, 360])
    else:
        orders = ('zyx', 'yxz', 'xzy', 'xyz', 'yzx', 'zxy',
                  'zyz', 'zxz', 'yzy', 'yxy', 'xzx', 'xyx')
        arr = np.deg2rad([-1.0, -0.1, -1e-3, 0, 1e-3, 0.1, 1.0])
        clusters = np.deg2rad([-360, -270, -180, -135, -90, -60, -45, -30,
                               0,
                               30, 45, 60, 90, 135, 180, 270, 360])

    angs = np.concatenate([center + arr for center in clusters])
    angs3 = np.dstack(np.meshgrid(angs, angs, angs))
    angs3 = angs3.reshape(-1, 3)

    fwd_funcs = (angle2rot, angle2dcm, eul2rot, eul2dcm)
    rev_funcs = (rot2angle, dcm2angle, rot2eul, dcm2eul)
    fwd_funcs = (eul2dcm, )
    rev_funcs = (dcm2eul, )
    for fn_ang2mat, fn_mat2ang in zip(fwd_funcs, rev_funcs):
        _print("Checking", fn_mat2ang.__name__)
        for order in orders:
            _print(order, '... ', sep='', end='')

            R0 = fn_ang2mat(angs3, order)
            x = fn_mat2ang(R0, order)
            R1 = fn_ang2mat(x, order)

            matched = np.isclose(R0, R1, atol=1e-15, rtol=1e-15)
            matched_mat = np.all(np.all(matched, axis=2), axis=1)

            # calculate relative difference between R0 and R1, which should
            # be equal up to round off - but note that the roundoff error
            # is affected by dividing by small numbers
            with np.errstate(divide='ignore', invalid='ignore'):
                max_R0_R1 = np.max([np.abs(R0), np.abs(R1)], axis=0)
                rel_diff = np.abs(R0 - R1) / max_R0_R1
                # mask out small numbers (the matrices are ortho-NORMAL,
                # so small is relative to unity)
                rel_diff[np.bitwise_and(np.abs(R0) < 1e-10,
                                        np.abs(R1) < 1e-10)] = np.nan
            success = np.all(matched_mat)
            if success:
                _print("Success!")
            else:
                _print("Failed!  {0:.4}% of matrices mismatched"
                       "".format(100 * np.sum(~matched_mat) / matched_mat.shape[0]))
            _print("           relative diff; 50%: {0:.2g}  90%: {1:.2g}  "
                   "99.9%: {2:.2g}  100%: {3:.2g}"
                   "".format(*np.nanpercentile(rel_diff, [50, 90, 99.9, 100])))

            if not success:
                i = np.nanargmax(np.nanmax(np.nanmax(rel_diff, axis=2), axis=1))
                _print()
                _print("    Transform with worst relative difference:")
                _print()
                _print("      Original Angles: ", angs3[i])
                _print("      rot2euls:      ", x[i])
                _print()
                _print("      R0:")
                _print("      ", str(R0[i]).replace('\n', '\n       '))
                _print("      R1:")
                _print("      ", str(R1[i]).replace('\n', '\n       '))
                _print("      Relative Difference:")
                _print("      ", str(rel_diff[i]).replace('\n', '\n       '))
                ret += 1
    return ret

def _check_wxyz2rot():
    _print("Checking wxyz2rot")

    assert np.allclose(wxyz2rotm(-np.pi / 2, [0, 1, 1]),
                       eul2rotm([-np.pi / 2, -np.pi / 4, np.pi / 4]))

    fname = "axang2rotm.csv"
    try:
        import pandas as pd
        if os.path.isfile(fname):
            df = pd.read_csv(fname)
            mlb_ax = df.loc[:, 'ax0':'ax2'].values.reshape(-1, 3).astype(np.double)
            mlb_ang = df.loc[:, 'ang'].values.reshape(-1, 1).astype(np.double)
            mlb_rots = df.loc[:, 'R00':'R22'].values.reshape(-1, 3, 3).astype('f8')

            py_mats = wxyz2rot(mlb_ang, mlb_ax)
            assert np.allclose(py_mats, mlb_rots)

            # double check eul2rotm because we can
            mlb_eul = df.loc[:, 'eul0':'eul2'].values.reshape(-1, 3).astype('f8')
            assert np.allclose(mlb_rots, eul2rotm(mlb_eul))
        else:
            raise OSError
    except OSError:
        _print("'{0}' does not exist, skipping test\n"
               "    (it can be generated using Viscid/scratch/gen_rotations.m)"
               "".format(fname))
        return 0
    except ImportError:
        _print("Pandas not installed, skipping test")
        return 0

    return 0

def _check_rot2wxyz():
    _print("Checking rot2wxyz")

    vectors = np.array([[0.0, 0.0, 1.0],
                        [0.0, 1.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 1.0],
                        [1.0, 0.0, 1.0],
                        [1.0, 1.0, 0.0],
                        [0.0, 0.0, -1.0],
                        [0.0, -1.0, 0.0],
                        [-1.0, 0.0, 0.0],
                        [0.0, -1.0, -1.0],
                        [-1.0, 0.0, -1.0],
                        [-1.0, -1.0, 0.0],
                        [0.0, 1.0, -1.0],
                        [1.0, 0.0, -1.0],
                        [1.0, -1.0, 0.0],
                        [0.0, -1.0, 1.0],
                        [-1.0, 0.0, 1.0],
                        [-1.0, 1.0, 0.0],
                        ])
    ang_centers = np.array([-2 * np.pi, -np.pi, -np.pi / 2, 0.0,
                           np.pi /2, np.pi, 2 * np.pi]).reshape(-1, 1)
    angles = np.array([-0.1, -0.01, 0.0, 0.01, 0.1]).reshape(1, -1) + ang_centers
    angles = np.reshape(angles, -1)

    for _, ax in enumerate(vectors):
        Rmats = wxyz2rot(angles, ax)
        new_ang, new_ax = rot2wxyz(Rmats)
        new_Rmats = wxyz2rot(new_ang, new_ax)
        assert np.allclose(Rmats, new_Rmats, atol=2e-8, rtol=2e-8)

    return 0

def _check_matrix2quat(quick=False):
    _print("Checking rotation matrix to quaternion")

    if quick:
        arr = np.deg2rad([-1.0, -1e-3, 0, 1e-3, 1.0])
        clusters = np.deg2rad([-360, -270, -180, -135, -90, -60, 0,
                               30, 90, 135, 180, 270, 360])
    else:
        arr = np.deg2rad([-1.0, -0.1, -1e-3, 0, 1e-3, 0.1, 1.0])
        clusters = np.deg2rad([-360, -270, -180, -135, -90, -60, -45, -30,
                               0,
                               30, 45, 60, 90, 135, 180, 270, 360])

    angs = np.concatenate([center + arr for center in clusters])
    angs3 = np.dstack(np.meshgrid(angs, angs, angs))
    angs3 = angs3.reshape(-1, 3)

    R = eul2rotm(angs3)

    q_rotm = rotm2quat(R)
    R2 = quat2rotm(q_rotm)
    assert np.allclose(R, R2)

    q_dcm = dcm2quat(R)
    R2 = quat2dcm(q_dcm)
    assert np.allclose(R, R2)

    return 0

def _check_quat2matrix():
    _print("Checking quaternion to rotation matrix")

    fname = "rotm2quat.csv"
    try:
        import pandas as pd
        if os.path.isfile(fname):
            df = pd.read_csv(fname)
            mlb_rots = df.loc[:, 'R00':'R22'].values.reshape(-1, 3, 3).astype('f8')
            mlb_q_rotm = df.loc[:, 'q0_rotm':'q3_rotm'].values.reshape(-1, 4).astype('f8')
            mlb_q_dcm = df.loc[:, 'q0_dcm':'q3_dcm'].values.reshape(-1, 4).astype('f8')

            py_rotm = quat2rotm(mlb_q_rotm)
            # valid = np.all(np.all(np.isclose(mlb_rots, py_rotm), axis=2), axis=1)
            assert np.allclose(mlb_rots, py_rotm)

            py_dcm = quat2dcm(mlb_q_dcm)
            # valid = np.all(np.all(np.isclose(mlb_rots, py_dcm), axis=2), axis=1)
            assert np.allclose(mlb_rots, py_dcm)

            return 0
        else:
            raise OSError
    except OSError:
        _print("'{0}' does not exist, skipping test\n"
               "    (it can be generated using Viscid/scratch/gen_rotations.m)"
               "".format(fname))
        return 0
    except ImportError:
        _print("Pandas not installed, skipping test")
        return 0

def _check_quat2axang(quick=False):
    _print("Checking quat <-> axis/angle representations")
    if quick:
        arr = np.deg2rad([-1.0, -1e-3, 0, 1e-3, 1.0])
        clusters = np.deg2rad([-360, -270, -180, -135, -90, -60, 0,
                               30, 90, 135, 180, 270, 360])
    else:
        arr = np.deg2rad([-1.0, -0.1, -1e-3, 0, 1e-3, 0.1, 1.0])
        clusters = np.deg2rad([-360, -270, -180, -135, -90, -60, -45, -30,
                               0,
                               30, 45, 60, 90, 135, 180, 270, 360])

    angs = np.concatenate([center + arr for center in clusters])
    angs3 = np.dstack(np.meshgrid(angs, angs, angs))
    angs3 = angs3.reshape(-1, 3)

    R0 = eul2rotm(angs3)
    axang0 = rotm2axang(R0)
    q1 = axang2quat(axang0)
    axang1 = quat2axang(q1)
    assert np.allclose(R0, quat2rotm(q1), atol=1e-7, rtol=1e-7)
    assert np.allclose(axang2rotm(axang0), quat2rotm(q1))
    assert np.allclose(axang2rotm(axang0), axang2rotm(axang1),
                       atol=1e-7, rtol=1e-7)

    R0 = eul2rotm(angs3)
    w0, xyz0 = rotm2wxyz(R0)
    q1 = wxyz2quat(w0, xyz0)
    w1, xyz1 = quat2wxyz(q1)
    assert np.allclose(R0, quat2rotm(q1), atol=1e-7, rtol=1e-7)
    assert np.allclose(wxyz2rotm(w0, xyz0), quat2rotm(q1))
    assert np.allclose(wxyz2rotm(w0, xyz0), wxyz2rotm(w1, xyz1),
                       atol=1e-7, rtol=1e-7)

    return 0

def _check_angle_between():
    _print("Checking angle_between")
    assert np.allclose(angle_between([1, 0, 0], [0, 1, 0]), np.pi / 2)
    assert np.allclose(angle_between([1, 0, 0], [0, -1, 0]), np.pi / 2)
    return 0

def _check_a2b_rot():
    _print("Checking a2b_rot")

    vectors = np.array([[0.0, 0.0, 1.0],
                        [0.0, 1.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 1.0],
                        [1.0, 0.0, 1.0],
                        [1.0, 1.0, 0.0],
                        [0.0, 0.0, -1.0],
                        [0.0, -1.0, 0.0],
                        [-1.0, 0.0, 0.0],
                        [0.0, -1.0, -1.0],
                        [-1.0, 0.0, -1.0],
                        [-1.0, -1.0, 0.0],
                        [0.0, 1.0, -1.0],
                        [1.0, 0.0, -1.0],
                        [1.0, -1.0, 0.0],
                        [0.0, -1.0, 1.0],
                        [-1.0, 0.0, 1.0],
                        [-1.0, 1.0, 0.0], ])
    angles = np.linspace(-2 * np.pi, 2 * np.pi, 33)

    for _, a in enumerate(vectors):
        for _, b in enumerate(vectors):
            assert np.allclose(np.einsum('mij,j->mi',
                                         a2b_rot(a, b, roll=angles),
                                         a),
                               b)
    return 0

def _check_quatmul():
    _print("Checking quatmul")
    R = eul2rotm([[np.pi / 3, 0, 0],
                  [0, np.pi / 6, 0],
                  [0, 0, np.pi / 4],
                  [np.pi / 3, np.pi / 6, 0],
                  [np.pi / 6, np.pi / 3, 0],
                  [0, np.pi / 3, np.pi / 4],
                  [0, np.pi / 4, np.pi / 3],
                  ])
    vecs = np.array([[1.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0],
                     [0.0, 0.0, 1.0],
                     [1.0, 1.0, 0.0],
                     [0.0, 1.0, 1.0],
                     [1.0, 0.0, 1.0],
                     [1.0, 1.0, 1.0],
                     [-1.0, 0.0, 0.0],
                     [0.0, -1.0, 0.0],
                     [0.0, 0.0, -1.0],
                     [-1.0, -1.0, 0.0],
                     [0.0, -1.0, -1.0],
                     [-1.0, 0.0, -1.0],
                     [-1.0, -1.0, -1.0],
                     ])

    def _run_test():
        q = rotm2quat(R)

        for vec in vecs:
            a = rotate(vec, *R)
            b = quat_rotate(vec, *q)
            assert np.allclose(a, b, atol=1e-14, rtol=1e-14)

        a = rotate(vecs, *R)
        b = quat_rotate(vecs, *q)
        assert np.allclose(a, b, atol=1e-14, rtol=1e-14)

    _run_test()

    try:
        import quaternion
        _run_test()
    except ImportError:
        _print("quaternion library not found, skipping test")

    return 0

def _check_all(quick=False):
    ret = 0
    ret += _check_return_shapes()
    ret += _check_wxyz2rot()
    ret += _check_rot2wxyz()
    ret += _check_angle_between()
    ret += _check_a2b_rot()

    ret += _check_matrix2quat(quick=quick)
    ret += _check_quat2matrix()
    ret += _check_quat2axang(quick=quick)

    ret += _check_quatmul()

    ret += _check_angle2matrix()
    ret += _check_matrix2angle(quick=quick)

    return ret

def _main():
    ret = _check_all()

    if ret == 0:
        _print("All tests succeeded!")
    else:
        _print("Some tests failed")

    return ret

if __name__ == "__main__":
    sys.exit(_main())

##
## EOF
##
