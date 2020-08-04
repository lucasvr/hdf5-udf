CXX        = g++
LDFLAGS    = -lhdf5
CXXFLAGS   = -O3 -Wall

CREATE_BIN = createh5
CREATE_SRC = createh5.cpp
CREATE_OBJ = $(patsubst %.cpp,%.o, $(CREATE_SRC))
READ_BIN   = readh5
READ_SRC   = readh5.cpp
READ_OBJ   = $(patsubst %.cpp,%.o, $(READ_SRC))

all: $(CREATE_BIN) $(READ_BIN)

files: $(CREATE_BIN)
	./$(CREATE_BIN) example-simple_vector.h5
	./$(CREATE_BIN) example-sine_wave.h5
	./$(CREATE_BIN) example-socket.h5
	./$(CREATE_BIN) example-add_datasets.h5 2

clean:
	rm -f $(CREATE_BIN) $(READ_BIN) *.o

$(CREATE_BIN): $(CREATE_OBJ)
	$(CXX) $^ -o $@ $(LDFLAGS)

$(READ_BIN): $(READ_OBJ)
	$(CXX) $^ -o $@ $(LDFLAGS)

$(OBJS):
