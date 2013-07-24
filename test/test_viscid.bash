#!/usr/bin/env bash

sdir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

tests="$( ls ${sdir}/test*.py )"
total=0
failed=0
for test in ${tests}; do
  echo "Running ${test}"
  ${test} "$@"
  status=$?
  if [ $status -eq 0 ]; then
    echo "Success!"
  else
    echo "Failed :("
    failed=$((failed+1))
  fi
  total=$((total+1))
  echo ""
done

if [ $failed -eq 0 ]; then
  echo "All tests passed! :)"
else
  echo "$failed of $total tests failed"
fi
