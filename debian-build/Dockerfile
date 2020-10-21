# First, build the prerequisite package syscall_intercept.
FROM ubuntu:20.04 AS syscall-builder

# Install prerequisites and clone syscall_intercept.
WORKDIR /root
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y clang cmake git devscripts libcapstone-dev pkg-config debhelper
RUN git clone https://github.com/pmem/syscall_intercept.git syscall-intercept

# Override the debian rules for syscall_intercept to:
# - Make sure we don't build test. They sometimes fail anyways.
# - Make sure we use clang.

# Make a dummy commit so that their build script consumes the new rules file correctly.
WORKDIR /root/syscall-intercept
RUN /usr/bin/echo -e "\noverride_dh_auto_configure:\n\tdh_auto_configure -- -DBUILD_TESTS=0 -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang" >> ./debian/rules
RUN git config user.email "dummy@example.com" && git config user.name "dummy" && git add ./debian/rules && git commit -m "Dummy commit"
RUN ./utils/build-deb.sh

# Now build the hdf-ud5 package.
FROM ubuntu:20.04 AS hdf-ud5-builder

ARG GIT_URL=https://github.com/lucasvr/hdf5-udf

# Install any prerequisite packages.
RUN mkdir /prereqs
COPY --from=syscall-builder /root/syscall-intercept/build-deb/*.deb /prereqs/
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential libhdf5-dev python3 \
    python3-dev python3-cffi libseccomp-dev luajit libluajit-5.1-dev pkg-config \
    git libcapstone-dev devscripts debhelper
RUN dpkg -i /prereqs/*.deb

# Clone code
RUN mkdir -p /root/build-deb
RUN git clone ${GIT_URL} /root/build-deb/hdf5-udf

# Do the actual build.
WORKDIR /root/build-deb/hdf5-udf
RUN debuild -us -uc

# Copy artifacts into sandbox container.
FROM alpine:latest
COPY --from=syscall-builder /root/syscall-intercept/build-deb /root/syscall_intercept
COPY --from=hdf-ud5-builder /root/build-deb /root/hdf5-udf

ENTRYPOINT ["/bin/ash"]
