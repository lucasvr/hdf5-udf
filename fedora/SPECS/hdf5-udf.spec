#%%define pre_dot   .alpha
#%%define pre_slash -alpha

Name:           hdf5-udf
Version:        1.0
Release:        1%{?pre_dot}%{?dist}
Summary:        Support for programmable datasets in HDF5 using Lua, Python, and C/C++ 

License:        MIT
URL:            https://github.com/lucasvr/hdf5-udf
Source0:        https://github.com/lucasvr/hdf5-udf/archive/v%{version}%{?pre_slash}/%{name}-%{version}%{?pre_slash}.tar.gz
Source1:        hdf5.pc

BuildRequires:  make gcc pkg-config
Requires:       capstone-devel libseccomp-devel luajit-devel hdf5-devel python3-devel gcc-c++
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
export PKG_CONFIG_PATH=${PKG_CONFIG_PATH}:${RPM_SOURCE_DIR}
make %{?_smp_mflags}

%install
rm -rf $RPM_BUILD_ROOT
make install DESTDIR="%{?buildroot}/usr" INSTALL="/usr/bin/install -p"

%files
%{_bindir}/%{name}
%{_datadir}/%{name}/*
%{_usr}/hdf5/lib/plugin/libhdf5-udf.so

%changelog
* Thu Oct 08 2020 Lucas C. Villa Real <lucasvr@gobolinux.org>
- First version
