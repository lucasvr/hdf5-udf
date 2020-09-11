DESTDIR        = /usr/local

CXX            = g++
CXXFLAGS       = $(shell pkg-config --cflags luajit) \
                 $(shell python3-config --includes) \
                 -fPIC -I. -Wall -O3 -g -ggdb
LDFLAGS        = $(shell pkg-config --libs luajit) \
                 $(shell python3-config --ldflags --embed) \
                 -ldl -lm -lhdf5 -Wl,--no-undefined

COMMON_SOURCES = backend.cpp lua_backend.cpp cpp_backend.cpp python_backend.cpp dataset.cpp miniz.cpp

FILTER_TARGET  = libhdf5-udf.so
FILTER_SOURCES = $(COMMON_SOURCES) hdf5-udf.cpp
FILTER_OBJS    = $(patsubst %.cpp,%.o, $(FILTER_SOURCES))
FILTER_LDFLAGS = -shared

BIN_TARGETS    = hdf5-udf
BIN_SOURCES    = $(COMMON_SOURCES) main.cpp
BIN_OBJS       = $(patsubst %.cpp,%.o, $(BIN_SOURCES))
BIN_CXXFLAGS   = -Wall


all: $(FILTER_TARGET) $(BIN_TARGETS)

clean:
	rm -f $(BIN_TARGETS) $(FILTER_TARGET) *.o

install:
	@install -v -d $(DESTDIR)/bin $(DESTDIR)/share/hdf5-udf $(DESTDIR)/hdf5/lib/plugin
	@install -v -t $(DESTDIR)/bin $(BIN_TARGETS)
	@install -v -t $(DESTDIR)/share/hdf5-udf udf_template.lua udf_template.cpp udf_template.py
	@install -v -t $(DESTDIR)/hdf5/lib/plugin $(FILTER_TARGET)

$(FILTER_TARGET): $(FILTER_OBJS)
	$(CXX) $^ -C -o $@ $(FILTER_LDFLAGS) $(LDFLAGS)

hdf5-udf: $(BIN_OBJS)
	$(CXX) $^ -o $@ $(LDFLAGS)

$(BIN_OBJS) $(FILTER_OBJS):
