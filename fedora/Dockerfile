FROM fedora:32
LABEL maintainer="lucasvr@gobolinux.org"

ARG github_tag=2.0
ARG syscall_intercept_basename=syscall_intercept-0.1

# Build dependencies (clang is needed to build syscall_intercept)
RUN dnf makecache -y
RUN dnf install -y make cmake clang gcc pkg-config git rpm-build

# Runtime dependencies
RUN dnf install -y capstone-devel libseccomp-devel luajit-devel hdf5-devel python3-devel gcc-c++

# Build time!
RUN mkdir -p /root/rpmbuild/{BUILD,RPMS,SOURCES,SPECS,SRPMS}
WORKDIR /root/rpmbuild
COPY fedora/SOURCES/* SOURCES/
COPY fedora/SPECS/* SPECS/

# Build syscall_intercept until an official package is provided
RUN git clone https://github.com/pmem/syscall_intercept.git && \
    mv syscall_intercept ${syscall_intercept_basename} && \
    tar czf SOURCES/${syscall_intercept_basename}.tar.gz ${syscall_intercept_basename} && \
    rm -rf ${syscall_intercept_basename}
RUN rpmbuild -ba SPECS/syscall_intercept.spec
RUN rpm -ivh RPMS/x86_64/*.rpm

# Build HDF5-UDF
RUN curl -L https://github.com/lucasvr/hdf5-udf/archive/${github_tag}.tar.gz \
         -o SOURCES/hdf5-udf-$(echo $github_tag | cut -b2-).tar.gz
RUN rpmbuild -ba SPECS/hdf5-udf.spec

ENTRYPOINT ["/bin/bash"]
