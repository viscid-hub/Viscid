#!/usr/bin/env bash
#
# It's super annoying having so many automated commits just
# to get the automatic doc generation onto gh-pages. Since
# we probably don't care about the history of these automated
# commits on the gh-pages branch, we can use this script to
# periodically squash and force push the gh-pages branch.

set -e

if [ "${1}" == "--pretend" -o "${1}" == "pretend" -o "${1}" == "-p" ]; then
  pretend=1
else
  pretend=0
fi

dir0="${PWD}"
sdir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
root_dir="${sdir}"
ghp_dir="${root_dir}/.ghpages"

squash_sha_cache=".ghp_squash_sha"

# Important, whoever is running this script must have ssh push
# access to the repository, and ideally they should know what
# they are doing... this script will do a soft reset and ammend
# a commmit followed by a force push upstream.
repo="git@github.com:KristoforMaynard/Viscid.git"

# make sure we're on the dev branch
cd ${root_dir}
current_branch="$(git rev-parse --abbrev-ref HEAD)"
if [ ${current_branch} != "dev" ]; then
  echo "Error: Squashing gh-pages must be done on the dev branch" >&2
  cd ${dir0}
  exit 1
fi

# clone the gh-pages branch separately
git clone -b gh-pages ${repo} "${ghp_dir}"
cd "${ghp_dir}"

squash_sha="$(cat ../${squash_sha_cache})"

if [ "$(git rev-parse HEAD)" == "${squash_sha}" ]; then
  echo "Info: gh-pages is already squashed" >&2
  cd ${root_dir}
  rm -rf ${ghp_dir}
  cd ${dir0}
  exit 0
fi

echo "Info: Soft resetting gh-pages branch" >&2
git reset --soft ${squash_sha}
echo "Info: Ammending gh-pages branch" >&2
git commit --amend -a -m "squashed docs / summaries"

new_squash_sha="$(git rev-parse HEAD)"

if [ ${pretend} -eq 0 ]; then
  echo "Info: Force push gh-pages branch to origin" >&2
  git push --force origin
else
  git log -n 3 >> ../DEBUG_LOG.txt
fi

cd ${root_dir}
rm -rf "${ghp_dir}"

# save the current state
has_changes=$(git diff-index --quiet HEAD --; echo $?)

if [ ${has_changes} -ne 0 ]; then
  echo "Info: Stashing current changes" >&2
  git stash save
else
  echo "Info: No current changes to stash" >&2
fi

# update && commit the squash_sha_cache
echo "${new_squash_sha}" > ${squash_sha_cache}
if [ ${pretend} -eq 0 ]; then
  echo "Info: Commiting updated gh-pages sha" >&2
  git add ${squash_sha_cache}
  git commit -m "update reference to squashed gh-pages"
else
  git reset --hard HEAD
fi

# put back previous changes
if [ ${has_changes} -ne 0 ]; then
  echo "Info: Resurrecting stashed changes" >&2
  git stash pop --index

  if [ $? -ne 0 ]; then
    echo "Warning: Your previous changes didn't apply, they should be the" >&2
    echo "         top-most stash." >&2
  fi
fi

# return us to wherever we were before all this nonsense happened
cd "${dir0}"
exit 0

##
## EOF
##
