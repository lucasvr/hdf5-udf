CREATE_BIN = ../build/createh5
READ_BIN   = ../build/readh5

files:
	$(CREATE_BIN) --out=example-compound.h5
	$(CREATE_BIN) --out=example-simple_vector.h5 --datatype=INT32
	$(CREATE_BIN) --out=example-sine_wave.h5 --datatype=INT32
	$(CREATE_BIN) --out=example-socket.h5 --datatype=INT32
	$(CREATE_BIN) --out=example-doom.h5 --datatype=INT32
	$(CREATE_BIN) --out=example-add_datasets.h5 --datatype=INT32 --count=2
	$(CREATE_BIN) --out=example-string.h5 --datatype=STRING --count=1
	$(CREATE_BIN) --out=example-varstring.h5 --datatype=VARSTRING --count=1
	$(CREATE_BIN) --out=example-multistring.h5 --datatype=VARSTRING --count=2
	$(CREATE_BIN) --out=example-compound-nostring_simple.h5 --compound=NOSTRING_SIMPLE --count=1
	$(CREATE_BIN) --out=example-compound-string_simple.h5 --compound=STRING_SIMPLE --count=1
	$(CREATE_BIN) --out=example-compound-varstring_simple.h5 --compound=VARSTRING_SIMPLE --count=1
	$(CREATE_BIN) --out=example-compound-nostring_mixed.h5 --compound=NOSTRING_MIXED --count=1
	$(CREATE_BIN) --out=example-compound-string_mixed.h5 --compound=STRING_MIXED --count=1
	$(CREATE_BIN) --out=example-compound-varstring_mixed.h5 --compound=VARSTRING_MIXED --count=1
	$(CREATE_BIN) --out=example-compound-varstring_mixed_plus_string.h5 --datatype=VARSTRING --compound=VARSTRING_MIXED --count=1
