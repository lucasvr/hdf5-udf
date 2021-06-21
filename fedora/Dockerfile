FROM fedora:32
LABEL maintainer="lucasvr@gobolinux.org"

ARG github_tag=2.1

# Build dependencies
RUN dnf makecache -y
RUN dnf install -y make clang gcc meson pkg-config git rpm-build

# Runtime dependencies
RUN dnf install -y libseccomp-devel libsodium-devel pcre-devel pcre-cpp luajit-devel hdf5-devel python3-devel gcc-c++

# Utilities needed by the RPM packing script
RUN dnf install -y which

# Build time!
RUN mkdir -p /root/rpmbuild/{BUILD,RPMS,SOURCES,SPECS,SRPMS}
WORKDIR /root/rpmbuild
COPY fedora/SOURCES/* SOURCES/
COPY fedora/SPECS/* SPECS/

RUN curl -L https://github.com/lucasvr/hdf5-udf/archive/${github_tag}.tar.gz \
         -o SOURCES/hdf5-udf-$(echo $github_tag | cut -b2-).tar.gz
RUN rpmbuild -ba SPECS/hdf5-udf.spec

ENTRYPOINT ["/bin/bash"]
