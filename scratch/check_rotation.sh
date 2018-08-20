#!/usr/bin/env bash

script_dir="$(dirname "${0}")"

cd "${script_dir}"

matlab -nodisplay -r "dump_rotations; exit"
python ../viscid/rotation.py
exit $?

