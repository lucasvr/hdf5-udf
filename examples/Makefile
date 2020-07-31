CXX        = g++
LDFLAGS    = -lhdf5
CXXFLAGS   = -O3 -Wall

TARGET     = createh5
SOURCES    = createh5.cpp
OBJS       = $(patsubst %.cpp,%.o, $(SOURCES))

all: $(TARGET)

files: $(TARGET)
	./$(TARGET) simple_vector.h5
	./$(TARGET) sine_wave.h5
	./$(TARGET) socket.h5
	./$(TARGET) add_datasets.h5 2

clean:
	rm -f $(TARGET) *.o

$(TARGET): $(OBJS)
	$(CXX) $^ -o $@ $(LDFLAGS)

$(OBJS):
