
export SHELL := /bin/bash

.PHONY: inplace build install clean test insttest flake8 refupdate deploy-html-ghpages doc-html doc-clean

# flake codes come from: http://flake8.readthedocs.org/en/latest/warnings.html
#                        http://pep8.readthedocs.org/en/latest/intro.html#error-codes
flake_on := E101,E111,E112,E113,E304
flake_off := E121,E123,E124,E126,E129,E133,E201,E202,E203,E226,E24,E266,E3,E402,E704
flake_off := $(flake_off),F401,F403,F841,C,N

all: build

dev:
	# shortcut for ./setup.py build_ext -i --with-cython
	python ./setup.py dev ${BUILD_ARGS}

inplace:
	python ./setup.py build_ext -i ${BUILD_ARGS}

build:
	python ./setup.py build_ext ${BUILD_ARGS}

install:
	python ./setup.py install ${BUILD_ARGS}

clean:
	python ./setup.py clean -a ${BUILD_ARGS}

check: test
test: inplace public-doccheck
	@echo "Note: Running tests on refreshed inplace build."
	VISCID_TEST_INPLACE=1 bash tests/viscid_runtests ${CHECK_ARGS}

instcheck: insttest
insttest: public-doccheck
	@echo "Note: Running tests using first Viscid in PYTHONPATH. Build was not"
	@echo "      implicitly refreshed."
	@echo "PYTHONPATH = ${PYTHONPATH}"
	VISCID_TEST_INPLACE=0 bash tests/viscid_runtests ${CHECK_ARGS}

flake8:
	flake8 --max-line-length=92 --show-source --select $(flake_on) \
	       --ignore $(flake_off) viscid tests scripts ${FLAKE_ARGS}

# update all reference plots
update_ref_plots: refupdate
update_ref: refupdate
refupdate:
	bash tests/viscid_update_ref_plots ${DOC_ARGS}

deploysummary: deploy-summary-ghpages
deploy-summary: deploy-summary-ghpages
deploy-summary-ghpages:
	bash tests/deploy_test_summary.sh ${DOC_ARGS}

deployhtml: deploy-html-ghpages
deploy-html: deploy-html-ghpages
deploy-html-ghpages:
	make ARGS="${DOC_ARGS}" -C doc deploy-html

html: doc-html
dochtml: doc-html
doc-html:
	make ARGS="${DOC_ARGS}" -C doc html

docclean: doc-clean
doc-clean:
	make ARGS="${DOC_ARGS}" -C doc clean


public-doccheck:
	make -C doc public-doccheck

