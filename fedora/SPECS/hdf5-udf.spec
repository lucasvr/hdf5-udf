#%%define pre_dot   .alpha
#%%define pre_slash -alpha

# 'strip' deletes the ELF section that holds the sandbox shared library, so disable it.
%define debug_package %{nil}

Name:           hdf5-udf
Version:        2.0
Release:        1%{?pre_dot}%{?dist}
Summary:        Support for programmable datasets in HDF5 using Lua, Python, and C/C++ 

License:        MIT
URL:            https://github.com/lucasvr/hdf5-udf
Source0:        https://github.com/lucasvr/hdf5-udf/archive/v%{version}%{?pre_slash}/%{name}-%{version}%{?pre_slash}.tar.gz
Source1:        hdf5.pc

BuildRequires:  meson gcc pkg-config
Requires:       libseccomp-devel libsodium-devel pcre-devel pcre-cpp luajit-devel hdf5-devel python3-devel gcc-c++ meson
ExclusiveArch:  x86_64

%description
HDF5-UDF is a mechanism to dynamically generate HDF5 datasets through user-defined functions
(UDFs) written in Lua, Python, or C/C++.

User-defined functions are compiled into executable form and the result is embedded into HDF5.
A supporting library gives access to existing datasets from user code so that data analysis and
derivation of other data can be produced.

%prep
%setup -q -n %{name}-%{version}%{?pre_slash}

%build
%meson -Dwith-python=true -Dwith-lua=true -Dwith-cpp=true --libexecdir=local/hdf5/lib/plugin
%meson_build

%install
%meson_install

%files
%{_bindir}/%{name}
%{_libdir}/lib%{name}.*
%{_libdir}/pkgconfig/%{name}.pc
%{_includedir}/%{name}.h
%{_usr}/local/hdf5/lib/plugin/libhdf5-udf-iofilter.so

%changelog
* Wed May 12 2021 Lucas C. Villa Real <lucasvr@gobolinux.org>
- Update build system

* Thu Oct 08 2020 Lucas C. Villa Real <lucasvr@gobolinux.org>
- First version
