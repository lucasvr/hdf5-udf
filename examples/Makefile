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
	./$(CREATE_BIN) --out=example-compound.h5 --compound --count=1

clean:
	rm -f $(CREATE_BIN) $(READ_BIN) *.o

$(CREATE_BIN): $(CREATE_OBJ)
	$(CXX) $^ -o $@ $(LDFLAGS)

$(READ_BIN): $(READ_OBJ)
	$(CXX) $^ -o $@ $(LDFLAGS)

$(OBJS):
