
if [[ "${SHELL}" == "/bin/zsh" ]]; then
  setopt shwordsplit
fi

pyvers="2.7 3.5 3.6 3.7"
for pyver in $pyvers; do
  pyver_nodot="${pyver//.}"

  conda_envname="vtest${pyver_nodot}conda"
  pip_envname="vtest${pyver_nodot}pip"

  conda create -qy -n ${conda_envname} python=${pyver}
  conda activate ${conda_envname}
  conda install -qy -c viscid-hub mayavi
  conda install -qy -c viscid-hub/label/dev viscid
  echo "---------------------------------"
  echo "Checking ${conda_envname}"
  echo "---------------------------------"
  python -m viscid check
  conda deactivate

  conda create -qy -n ${pip_envname} python=${pyver} pip
  conda activate ${pip_envname}
  python -m pip install numpy matplotlib
  python -m pip install viscid==1.0.0.dev14
  echo "---------------------------------"
  echo "Checking ${pip_envname}"
  echo "---------------------------------"
  python -m viscid check
  conda deactivate
done
