FC=gfortran
F2PY := $(shell [ $(which f2py3) ] && echo "f2py3" || echo "f2py")

%.f: %.for
	fppn -c -- $< >$@

%.f90: %.for
	fppn -c -- $< >$@

.F.o:
	$(FC) -c -o $@ $>

tracer: tracer.f90
	# $(FC) -o $@ $^
	$(F2PY) -c -m $@ $^

all: tracer

clean:
	rm -f *.o *.so
