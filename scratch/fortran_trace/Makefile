
FC=gfortran
F2PY=f2py-2.7

%.f: %.for
	fppn -c -- $< >$@

%.f90: %.for
	fppn -c -- $< >$@

.F.o:
	$(FC) -c -o $@ $>

tracer: tracer.f90
	# $(FC) -o $@ $^
	$(F2PY) -c -m $@ $^

for_bench: for_bench.f90
	# $(FC) -o $@ $^
	$(F2PY) -c -m $@ $^

all: tracer for_bench

clean:
	rm -f *.o *.so
