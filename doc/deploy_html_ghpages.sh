#!/usr/bin/env bash
# Deploy documents to gh-pages. This should be called through
# the Makefile as
#   > make doc-deploy

set -e

dir0=${PWD}
sdir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${sdir}

doc_dir="${PWD}"
build_dir="${doc_dir}/_build"
html_dir="${build_dir}/html"
root_dir="${doc_dir}/../"

# use the part of the branch that is after the last '/', ie, in
# "feature/wizbang", only use "wizbang"
if [[ -n ${TRAVIS} && -n ${CONTINUOUS_INTEGRATION} ]]; then
  branch=${TRAVIS_BRANCH}
  tags=${TRAVIS_TAG}
else
  branch="$(git rev-parse --abbrev-ref HEAD)"
  branch="${branch##*/}"
  tags="$(echo -n $(git tag -l --contains HEAD) | tr -s ' ' ':')"
fi

if [ ${branch} == "HEAD" ]; then
  echo "Oops: Can't deploy a detached HEAD"
  exit 1
fi

# assert that the branch name is in the version number... this
# keeps things from getting confusing on pages, ie, the question
# 'to which version of the code is this page relevant'
set +e
echo "> branch = ${branch}"
if [ "${branch}" == "master" ]; then
  grep "__version__" ${root_dir}/viscid/__init__.py | grep -v dev | grep -v travis
  if [ $? -ne 0 ]; then
    echo "ERROR: Can only deploy to gh-pages if the branch name is in the version."
    exit 1
  fi
else
  grep "__version__" ${root_dir}/viscid/__init__.py | grep "${branch}"
  if [ $? -ne 0 ]; then
    echo "ERROR: Can only deploy to gh-pages if the branch name is in the version."
    exit 1
  fi
fi

set -e

# Make sure Viscid is built inplace, and make the docs with the inplace
# build at the top of the PYTHONPATH
make -C .. inplace
export PYTHONPATH="${root_dir}:${PYTHONPATH}"
make clean
make html

msg="Automatic doc update ${branch}:${tags}"
${root_dir}/deploy_ghpages -cd "docs/${branch}"                \
                           -r "KristoforMaynard/Viscid-docs"   \
                           -m "${msg}"                         \
                           "${html_dir}"/*

##
## EOF
##
