##################################
# configuration and build options
##################################

# OPT_SANDBOX: 0=disable syscall filtering, 1=enable
OPT_SANDBOX   := 1

DESTDIR        = /usr/local

CXX            = g++
OBJCOPY        = objcopy
CXXFLAGS       = $(shell pkg-config --cflags luajit) \
                 $(shell python3-config --includes) \
                 -fPIC -I. -Wall -O3 -g -ggdb
LDFLAGS        = $(shell pkg-config --libs luajit) \
                 $(shell python3-config --ldflags --embed) \
                 -ldl -lm -lhdf5 -Wl,--no-undefined

ALL_HEADERS    = $(wildcard *.h)
COMMON_SOURCES = backend.cpp lua_backend.cpp cpp_backend.cpp \
                 python_backend.cpp dataset.cpp miniz.cpp

######################
# libhdf5-udf-sandbox
######################

ifeq ($(OPT_SANDBOX),1)
CXXFLAGS       += -DENABLE_SANDBOX
COMMON_SOURCES +=  sandbox.cpp

SANDBOX_TARGET  = libhdf5-udf-sandbox.so
SANDBOX_SOURCES = sandbox_library.cpp
SANDBOX_OBJS    = $(patsubst %.cpp,%.o, $(SANDBOX_SOURCES))
SANDBOX_LDFLAGS = -shared -fPIC -lsyscall_intercept -lseccomp
endif

##############
# libhdf5-udf
##############

FILTER_TARGET  = libhdf5-udf.so
FILTER_SOURCES = $(COMMON_SOURCES) hdf5-udf.cpp
FILTER_OBJS    = $(patsubst %.cpp,%.o, $(FILTER_SOURCES))
FILTER_LDFLAGS = -shared

###########
# hdf5-udf
###########

BIN_TARGET     = hdf5-udf
BIN_SOURCES    = $(COMMON_SOURCES) main.cpp
BIN_OBJS       = $(patsubst %.cpp,%.o, $(BIN_SOURCES))
BIN_CXXFLAGS   = -Wall

################
# general rules
################

all: $(SANDBOX_TARGET) $(FILTER_TARGET) $(BIN_TARGET) $(SANDBOX_TARGET)

$(SANDBOX_TARGET): $(SANDBOX_OBJS)
	$(CXX) $^ -C -o $@ $(SANDBOX_LDFLAGS)

$(FILTER_TARGET): $(FILTER_OBJS)
	$(CXX) $^ -C -o $@ $(FILTER_LDFLAGS) $(LDFLAGS)
	@if [ ! -z "$(SANDBOX_TARGET)" ]; then \
		$(OBJCOPY) \
		--add-section .hdf5-udf-sandbox=$(SANDBOX_TARGET) \
		--set-section-flags .hdf5-udf-sandbox=noload,readonly \
		$(FILTER_TARGET); \
	fi

$(BIN_TARGET): $(BIN_OBJS)
	$(CXX) $^ -o $@ $(LDFLAGS)

$(SANDBOX_OBJS) $(BIN_OBJS) $(FILTER_OBJS): $(ALL_HEADERS)

clean:
	rm -f $(BIN_TARGET) $(FILTER_TARGET) $(WRAPPER_TARGET) *.o

install:
	@install -v -d $(DESTDIR)/bin $(DESTDIR)/share/hdf5-udf $(DESTDIR)/hdf5/lib/plugin
	@install -v -t $(DESTDIR)/bin $(BIN_TARGET)
	@install -v -t $(DESTDIR)/share/hdf5-udf udf_template.{lua,cpp,py}
	@install -v -t $(DESTDIR)/hdf5/lib/plugin $(FILTER_TARGET)