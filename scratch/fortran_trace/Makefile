
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
	$(F2PY) -c -m tracer $^

all: tracer

clean:
	rm -f *.o *.so
