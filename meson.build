#######################
# Global build options
#######################

project('HDF5-UDF', ['c', 'cpp'],
  version: '1.2',
  default_options: [
    'warning_level=1',
    'cpp_std=c++14',
    'default_library=shared',
    'prefix=/usr/local',
    'libexecdir=hdf5/lib/plugin'])

sources = ['src/backend.cpp', 'src/dataset.cpp']
sandbox_deps = []
deps = []

########################
# Required dependencies
########################

deps += [
  meson.get_compiler('cpp').find_library('dl', required: true),
  dependency('hdf5', language: 'c', required: true),
  dependency('libsodium', required: true)
]

########################
# Optional dependencies
########################

if get_option('with-python')
  sources += ['src/python_backend.cpp']
  deps += [
    dependency('python3', required: true),
    dependency('python3-embed', required: false)
  ]
  add_global_arguments('-DENABLE_PYTHON', language: 'cpp')
endif

if get_option('with-cpp')
  sources += ['src/cpp_backend.cpp', 'src/miniz.cpp']
  deps += [meson.get_compiler('cpp').find_library('m', required: true)]
  add_global_arguments('-DENABLE_CPP', language: 'cpp')
endif

if get_option('with-lua')
  sources += ['src/lua_backend.cpp']
  deps += [dependency('luajit', required: true)]
  add_global_arguments('-DENABLE_LUA', language: 'cpp')
endif

if get_option('with-sandbox')
  sources += ['src/sandbox.cpp']
  seccomp_dep = dependency('libseccomp', required: true)
  syscall_intercept_dep = dependency('libsyscall_intercept', required: true)
  sandbox_deps += [seccomp_dep, syscall_intercept_dep]
  deps += [seccomp_dep]
  add_global_arguments('-DENABLE_SANDBOX', language: 'cpp')
endif

summary({
  'Python support': get_option('with-python'),
  'C/C++ support': get_option('with-cpp'),
  'Lua support': get_option('with-lua'),
  'Sandbox support': get_option('with-sandbox'),
}, section: 'Configuration')

######################
# Output main program
######################

executable(
  'hdf5-udf',
  sources + ['src/main.cpp', 'src/user_profile.cpp'],
  dependencies: deps,
  install: true)

#########################
# Output HDF5 I/O filter
#########################

iofilter_lib = library(
  'hdf5-udf',
  sources + ['src/io_filter.cpp', 'src/user_profile.cpp'],
  dependencies: deps,
  install_dir: get_option('libexecdir'),
  install: true)

#########################
# Output sandbox library
#########################

if get_option('with-sandbox')
  sandbox_lib = library(
    'hdf5-udf-sandbox',
    ['src/sandbox_library.cpp'],
    dependencies: sandbox_deps,
    install: false)

  custom_target('ELF section for sandbox',
    input: [sandbox_lib, iofilter_lib],
    output: ['dummy'],
    command: [
      join_paths(meson.source_root(), 'src/modify_elf_section'),
      iofilter_lib.full_path(),
      sandbox_lib.full_path()
    ],
    build_by_default: true)
endif

########################
# Install UDF templates
########################

install_data('src/udf_template.cpp', install_dir: get_option('datadir')/'hdf5-udf')
install_data('src/udf_template.lua', install_dir: get_option('datadir')/'hdf5-udf')
install_data('src/udf_template.py', install_dir: get_option('datadir')/'hdf5-udf')
