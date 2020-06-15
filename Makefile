all:
	+make -C src
	+make -C examples

clean:
	+make -C src clean
	+make -C examples clean

install:
	+make -C src install
