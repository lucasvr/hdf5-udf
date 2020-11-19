##################################
# configuration and build options
##################################

OPT_SANDBOX   := 1 # enable/disable syscall filtering
OPT_PYTHON    := 1 # enable/disable Python backend
OPT_LUA       := 1 # enable/disable Lua/LuaJIT backend
OPT_CPP       := 1 # enable/disable C/C++ backend

DESTDIR        =
prefix         = /usr/local
PLUGINDIR      = $(prefix)/hdf5/lib/plugin

# Starting with Python 3.8, `python3-config --ldflags` no longer links to libpython;
# that functionality is now provided by `python3-config --embed`, which didn't exist
# prior to 3.8.
ifeq ($(shell python3-config --help | grep embed),)
PYTHON_LDFLAGS = $(shell python3-config --ldflags)
else
PYTHON_LDFLAGS = $(shell python3-config --ldflags --embed)
endif

CXX            = g++
OBJCOPY        = objcopy
CXXFLAGS       = $(shell pkg-config --cflags luajit hdf5) \
                 $(shell python3-config --includes) \
                 -fPIC -I. -Wall -O3 -g -ggdb
LDFLAGS        = $(shell pkg-config --libs luajit hdf5) \
                 $(PYTHON_LDFLAGS) \
                 -ldl -lm -Wl,--no-undefined

ALL_HEADERS    = $(wildcard *.h)
COMMON_SOURCES = backend.cpp dataset.cpp

ifeq ($(strip $(OPT_PYTHON)),1)
CXXFLAGS       += -DENABLE_PYTHON
COMMON_SOURCES += python_backend.cpp
endif

ifeq ($(strip $(OPT_LUA)),1)
CXXFLAGS       += -DENABLE_LUA
COMMON_SOURCES += lua_backend.cpp
endif

ifeq ($(strip $(OPT_CPP)),1)
CXXFLAGS       += -DENABLE_CPP
COMMON_SOURCES += cpp_backend.cpp miniz.cpp
endif

######################
# libhdf5-udf-sandbox
######################

ifeq ($(strip $(OPT_SANDBOX)),1)
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

$(FILTER_TARGET): $(FILTER_OBJS) $(SANDBOX_TARGET)
	$(CXX) $(FILTER_OBJS) -C -o $@ $(FILTER_LDFLAGS) $(LDFLAGS)
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
	rm -f $(BIN_TARGET) $(FILTER_TARGET) $(SANDBOX_TARGET) $(WRAPPER_TARGET) *.o

install:
	@install -v -d $(DESTDIR)$(prefix)/bin $(DESTDIR)$(prefix)/share/hdf5-udf $(DESTDIR)$(PLUGINDIR)
	@install -v -t $(DESTDIR)$(prefix)/bin $(BIN_TARGET)
	@install -v -t $(DESTDIR)$(prefix)/share/hdf5-udf udf_template.lua
	@install -v -t $(DESTDIR)$(prefix)/share/hdf5-udf udf_template.cpp
	@install -v -t $(DESTDIR)$(prefix)/share/hdf5-udf udf_template.py
	@install -v -t $(DESTDIR)$(PLUGINDIR) $(FILTER_TARGET)
