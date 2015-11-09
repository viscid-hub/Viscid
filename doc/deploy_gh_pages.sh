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
ghp_dir="${root_dir}/.ghpages"

# use the part of the branch that is after the last '/', ie, in
# "feature/wizbang", only use "wizbang"
branch="$(git rev-parse --abbrev-ref HEAD)"
branch="${branch##*/}"
tags="$(echo -n $(git tag -l --contains HEAD) | tr -s ' ' ':')"

if [ ${branch} == "HEAD" ]; then
  echo "Can't deploy a detached HEAD"
  exit 1
fi

# assert that the branch name is in the version number... this
# keeps things from getting confusing on pages, ie, the question
# 'to which version of the code is this page relevant'
set +e
echo "> branch = ${branch}"
grep "__version__" ${root_dir}/viscid/__init__.py | grep "${branch}"
if [ "${branch}" != "master" ] && [ $? -ne 0 ]; then
  echo "ERROR: Can only deploy to gh-pages if the branch name is in the version."
  exit 1
fi
set -e

make clean
make html

if [ -n "$GH_TOKEN" ] && [ -n "$GH_REF" ]; then
  repo="https://${GH_TOKEN}@${GH_REF}"
  git config user.name "Travis-CI"
  git config user.email "nobody@travis-ci.org"
  git config push.default simple
elif if [ -n "$GH_REF" ]; then
  echo ">> There is a GH_REF, but not TOKEN, this is probably a pull request,"
  echo ">> and pull requests can't update the gh-pages"
  exit 0
else
  repo="git@github.com:KristoforMaynard/Viscid.git"
fi
echo "using repo::" ${repo}
git clone -b gh-pages ${repo} ${ghp_dir}

dest_dir=${ghp_dir}/docs/${branch}

if [ -d ${dest_dir} ]; then
  rm -rf ${dest_dir}
fi

mkdir -p ${dest_dir}
echo "" >> ${ghp_dir}/.nojekyll
cp -r ${html_dir}/* ${dest_dir}

cd ${ghp_dir}
git add ${dest_dir}
if [ "$(git diff --name-only --cached)" != "" ]; then
  git commit -m "Automatic doc update ${branch}:${tags}"
  git push
else
  echo "Docs didn't change"
fi

# cd ${doc_dir}
rm -rf ${ghp_dir}
cd ${dir0}
set +e

##
## EOF
##
