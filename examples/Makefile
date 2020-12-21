CXX        = g++
LDFLAGS    = $(shell pkg-config --libs hdf5)
CXXFLAGS   = $(shell pkg-config --cflags hdf5) -O3 -Wall

CREATE_BIN = createh5
CREATE_SRC = createh5.cpp
CREATE_OBJ = $(patsubst %.cpp,%.o, $(CREATE_SRC))
READ_BIN   = readh5
READ_SRC   = readh5.cpp
READ_OBJ   = $(patsubst %.cpp,%.o, $(READ_SRC))

all: $(CREATE_BIN) $(READ_BIN)

files: $(CREATE_BIN)
	./$(CREATE_BIN) --out=example-simple_vector.h5
	./$(CREATE_BIN) --out=example-sine_wave.h5
	./$(CREATE_BIN) --out=example-socket.h5
	./$(CREATE_BIN) --out=example-doom.h5
	./$(CREATE_BIN) --out=example-add_datasets.h5 --count=2
	./$(CREATE_BIN) --out=example-compound-nostring_simple.h5 --compound=NOSTRING_SIMPLE --count=1
	./$(CREATE_BIN) --out=example-compound-string_simple.h5 --compound=STRING_SIMPLE --count=1
	./$(CREATE_BIN) --out=example-compound-varstring_simple.h5 --compound=VARSTRING_SIMPLE --count=1
	./$(CREATE_BIN) --out=example-compound-nostring_mixed.h5 --compound=NOSTRING_MIXED --count=1
	./$(CREATE_BIN) --out=example-compound-string_mixed.h5 --compound=STRING_MIXED --count=1
	./$(CREATE_BIN) --out=example-compound-varstring_mixed.h5 --compound=VARSTRING_MIXED --count=1

clean:
	rm -f $(CREATE_BIN) $(READ_BIN) *.o

$(CREATE_BIN): $(CREATE_OBJ)
	$(CXX) $^ -o $@ $(LDFLAGS)

$(READ_BIN): $(READ_OBJ)
	$(CXX) $^ -o $@ $(LDFLAGS)

$(OBJS):
