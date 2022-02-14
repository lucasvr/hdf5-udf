#######################
# Global build options
#######################

project('HDF5-UDF', ['c', 'cpp'],
  version: '2.1',
  meson_version: '>= 0.46.0',
  default_options: [
    'warning_level=1',
    'cpp_std=c++14',
    'default_library=shared',
    'prefix=/usr/local',
    'libexecdir=hdf5/lib/plugin'])

sources = [
  'src/backend.cpp',
  'src/dataset.cpp',
  'src/libudf.cpp',
  'src/file_search.cpp',
  'src/user_profile.cpp',
  'src/os_linux.cpp',
  'src/os_macos.cpp',
  'src/os_posix.cpp',
  'src/os_windows.cpp'
]
seccomp_dep = ''
deps = []

if host_machine.system() == 'windows'
  deps += meson.get_compiler('cpp').find_library('secur32', required: true)
endif


#######################################
# Automatically generated dependencies
#######################################

file2hex = find_program('src/file_to_hex.py')
udf_template_cpp = custom_target(
  'udf_template_cpp',
  input: ['src/udf_template.cpp'],
  output: ['udf_template_cpp.h'],
  command: [file2hex, '@INPUT@', '@SOURCE_ROOT@/src/@OUTPUT@']
)
udf_template_cu = custom_target(
  'udf_template_cu',
  input: ['src/udf_template.cu'],
  output: ['udf_template_cu.h'],
  command: [file2hex, '@INPUT@', '@SOURCE_ROOT@/src/@OUTPUT@']
)
udf_template_lua = custom_target(
  'udf_template_lua',
  input: ['src/udf_template.lua'],
  output: ['udf_template_lua.h'],
  command: [file2hex, '@INPUT@', '@SOURCE_ROOT@/src/@OUTPUT@']
)
udf_template_py = custom_target(
  'udf_template_py',
  input: ['src/udf_template.py'],
  output: ['udf_template_py.h'],
  command: [file2hex, '@INPUT@', '@SOURCE_ROOT@/src/@OUTPUT@']
)

########################
# Required dependencies
########################

hdf5_lib = dependency('hdf5', language: 'c', required: true)
deps += [
  hdf5_lib,
  meson.get_compiler('cpp').find_library('dl', required: true),
  dependency('libsodium', required: true),
  dependency('libpcrecpp', required: true)
]

########################
# Optional dependencies
########################

if get_option('with-python')
  sources += ['src/backend_python.cpp']
  deps += [
    dependency('python3', required: true),
    dependency('python3-embed', required: false)
  ]
  add_global_arguments('-DENABLE_PYTHON', language: 'cpp')
endif

if get_option('with-cpp')
  sources += ['src/backend_cpp.cpp', 'src/miniz.cpp']
  deps += [meson.get_compiler('cpp').find_library('m', required: true)]
  add_global_arguments('-DENABLE_CPP', language: 'cpp')
endif

if get_option('with-lua')
  sources += ['src/backend_lua.cpp']
  deps += [dependency('luajit', required: true)]
  add_global_arguments('-DENABLE_LUA', language: 'cpp')
endif

if get_option('with-gds')
  sources += ['src/backend_gds.cpp', 'src/backend_gds_mm.cpp']
  cuda_libdir = '/usr/local/cuda/targets/x86_64-linux/lib'
  deps += [
    meson.get_compiler('cpp').find_library('cufile', dirs: cuda_libdir, required: true),
    meson.get_compiler('cpp').find_library('cudart', dirs: cuda_libdir, required: true),
    meson.get_compiler('cpp').find_library('udf_snappy_reader', required: true),
    meson.get_compiler('cpp').find_library('gomp', required: true)
  ]
  add_global_arguments('-fopenmp', language: 'cpp')
  add_global_arguments('-DENABLE_GDS', language: 'cpp')
  add_global_arguments('-I/usr/local/cuda/targets/x86_64-linux/include', language: 'cpp')
  add_global_arguments('-I/usr/include/snappy-cuda', language: 'cpp')
endif

if get_option('with-sandbox')
  if build_machine.system() == 'linux'
    sources += ['src/sandbox_linux.cpp']
    deps += [dependency('libseccomp', required: true)]
  elif build_machine.system() == 'darwin'
    sources += ['src/sandbox_macos.cpp']
  endif
  add_global_arguments('-DENABLE_SANDBOX', language: 'cpp')
endif

message('\n' +
'\n  HDF5-UDF: User-Defined Functions for HDF5\n' +
'\n  Configuration' +
'\n    Python support:     ' + get_option('with-python').to_string() +
'\n    C/C++ support:      ' + get_option('with-cpp').to_string() +
'\n    Lua support:        ' + get_option('with-lua').to_string() +
'\n    NVIDIA GDS support: ' + get_option('with-gds').to_string() +
'\n    Sandbox support:    ' + get_option('with-sandbox').to_string() +
'\n')

######################
# Output main library
######################

hdf5_udf_lib = shared_library(
  'hdf5-udf',
  sources + [udf_template_cpp, udf_template_cu, udf_template_lua, udf_template_py],
  version: '2.1',
  soversion: '0',
  dependencies: deps,
  install: true)

######################
# Output main program
######################

executable(
  'hdf5-udf',
  ['src/main.cpp'],
  link_with: hdf5_udf_lib,
  dependencies: deps,
  install: true)

#########################
# Output HDF5 I/O filter
#########################

iofilter_lib = shared_library(
  'hdf5-udf-iofilter',
  ['src/io_filter.cpp'],
  link_with: hdf5_udf_lib,
  dependencies: deps,
  install_dir: get_option('libexecdir'),
  install: true)

############################################
# Install headers, generate pkg-config file
############################################
pkg = import('pkgconfig')
pkg.generate(
  libraries: [seccomp_dep, hdf5_udf_lib],
  libraries_private: [],
  name: 'HDF5-UDF',
  version: '2.1',
  filebase: 'hdf5-udf',
  description: 'User-defined functions for HDF5'
)

install_headers(
  'src/hdf5-udf.h'
)

###################
# Output utilities
###################
executable(
  'createh5',
  ['examples/createh5.cpp'],
  dependencies: [hdf5_lib],
  install: false)

executable(
  'readh5',
  ['examples/readh5.cpp'],
  dependencies: [hdf5_lib],
  install: false)
