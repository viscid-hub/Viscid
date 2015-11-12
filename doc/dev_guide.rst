Developer's Guide
=================

Style
-----

Please, if you edit the code, use [PEP 8](https://www.python.org/dev/peps/pep-0008/) style. Poor style is more than just aesthetic; it tends to lead to bugs that are difficult to spot. This is sorely true when it comes to whitespace (4 spaces per indent please). Here are some aspects of PEP 8 that are ok to bend:

  + Lines should be less than 92 characters long, but if you're pushing over 80, then think about whether or not you're doing too much on one line.
  + Additional whitespace around parenthases and operators should be used to line up successive lines of math if that makes the math operation more clear. Note that this says additional, i.e., don't start dropping spaces after commas. An example is doing cross products; the code should be written such that it's visually clear that the lines calculating the components of the vector are cyclic permutations.

Continuous integration will pass commits and pull requests through [Flake8](https://flake8.readthedocs.org/en/latest/) with a generous amount of ignored rules. See `Viscid/Makefile` for the list of rules that are ignored.

Continuous Integration
----------------------

This project uses [Travis-CI](http://travis-ci.org) for continuous integration. By default, tests run whenever commits are pushed or pull requests are made to the main git repository on GitHub. Test summary pages are automatically uploaded to GitHub Pages in the `summary` directory. In additon, commits to `master` and `dev` automatically update the html documentation and push the changes to GitHub Pages.

  Forks can use Travis-CI by enabling it for their own GitHub accounts. The caveat is that commits to `master` and `dev` from forked repositories will not update GitHub Pages.

Git structure
-------------

This project uses the Git Flow merge strategy for development / release cycles. Basically, changes should be made on `feature/*` branches and merged into the `dev` branch when ready. Releases should begin on a branch of dev named `release/version-number` which is pushed upstream. The push is important because it triggers CI tests. When the release branch instills confidence, it can be merged into both `master` and `dev`.

Here are some things to remember when doing releases and the like:

  + The first commit to a `release` branch should remove the word "dev" from both `viscid.__version__` and `Viscid/CHANGES.md`. This is checked by Travis-CI.
  + The first commit to the `dev` branch after merging a `release` should be to bump the version number in both `viscid.__version__` and `Viscid/CHANGES.md`. Remember to put the word "dev" at the end of the version number. This is checked by Travis-CI.
  + There shouldn't be a need to explicitly generate the html docs since they're recreated and pushed to GitHub Pages on pushes to the dev and master branches on the mait GitHub repo.
