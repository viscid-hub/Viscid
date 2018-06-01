#!/usr/bin/env bash
# Deploy documents to gh-pages. This should be called through

set -e

sdir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
root_dir="${sdir}/../"

# use the part of the branch that is after the last '/', ie, in
# "feature/wizbang", only use "wizbang"
if [[ -n ${TRAVIS} && -n ${CONTINUOUS_INTEGRATION} ]]; then
  branch=${TRAVIS_BRANCH}
  tags=${TRAVIS_TAG}
  name="${branch}-${PYTHON}"
  if [[ "${DEPS}" != "" ]]; then
    name="${name}-${DEPS}"
  fi
else
  branch="$(git rev-parse --abbrev-ref HEAD)"
  branch="${branch##*/}"
  tags="$(echo -n $(git tag -l --contains HEAD) | tr -s ' ' ':')"
  name="${branch}"
fi


msg="Automatic summary upload ${branch}:${tags}"
${root_dir}/deploy_ghpages -d "summary/${name}"                               \
                           -m "${msg}"                                        \
                           ${sdir}/index.html ${sdir}/plots ${sdir}/ref_plots
echo "Summary page uploaded to http://kristoformaynard.github.io/Viscid-docs/summary/${name}"
