#######################
# Global build options
#######################

project('HDF5-UDF', ['c', 'cpp'],
  version: '2.0',
  meson_version: '>= 0.40.1',
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
  'src/user_profile.cpp'
]
syscall_intercept_dep = ''
seccomp_dep = ''
deps = []


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
  deps += [seccomp_dep]
  add_global_arguments('-DENABLE_SANDBOX', language: 'cpp')
endif

message('\n' +
'\n  HDF5-UDF: User-Defined Functions for HDF5\n' +
'\n  Configuration' +
'\n    Python support : ' + get_option('with-python').to_string() +
'\n    C/C++ support  : ' + get_option('with-cpp').to_string() +
'\n    Lua support    : ' + get_option('with-lua').to_string() +
'\n    Sandbox support: ' + get_option('with-sandbox').to_string() +
'\n')

######################
# Output main library
######################

hdf5_udf_lib = shared_library(
  'hdf5-udf',
  sources + [udf_template_cpp, udf_template_lua, udf_template_py],
  version: '2.0',
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

#########################
# Output sandbox library
#########################

if get_option('with-sandbox')
  sandbox_lib = shared_library(
    'hdf5-udf-sandbox',
    ['src/sandbox_library.cpp'],
    dependencies: [seccomp_dep, syscall_intercept_dep],
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

############################################
# Install headers, generate pkg-config file
############################################
pkg = import('pkgconfig')
pkg.generate(
  libraries: [seccomp_dep, hdf5_udf_lib],
  libraries_private: [syscall_intercept_dep],
  name: 'HDF5-UDF',
  version: '2.0',
  filebase: 'hdf5-udf',
  description: 'User-defined functions for HDF5'
)

install_headers(
  'src/hdf5-udf.h'
)