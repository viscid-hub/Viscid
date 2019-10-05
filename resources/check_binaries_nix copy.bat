
@echo off

for %%p in (2.7 3.5 3.6 3.7) do (
  conda create -qy -n vtestconda%%p python=%%p
  conda activate vtestconda%%p
  conda install -qy -c viscid-hub mayavi
  conda install -qy -c viscid-hub/label/dev viscid
  echo ---------------------------------
  echo Checking vtestconda%%p
  echo ---------------------------------
  python -m viscid check
  conda deactivate

  conda create -qy -n vtestpip%%p python=%%p pip
  conda activate vtestpip%%p
  python -m pip install numpy matplotlib
  python -m pip install viscid==1.0.0.dev14
  echo ---------------------------------
  echo Checking vtestpip%%p
  echo ---------------------------------
  python -m viscid check
  conda deactivate
)
