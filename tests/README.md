Guidelines for Tests
--------------------

+ Tests should be named `test_TESTNAME.py`
+ Tests should generate plots as `png` files that begin with the test name, ie, `test_ascii.py` will generate plots named `ascii_*.png`.
+ When plotting, `savefig` should always be called before `show`


Updating Reference Plots
------------------------

Use the `Viscid/tests/update_ref_plots` script.
