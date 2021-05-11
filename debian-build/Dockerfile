# Build the hdf-ud5 package.
FROM ubuntu:20.04 AS hdf-ud5-builder

ARG GIT_URL=https://github.com/lucasvr/hdf5-udf

# Install any prerequisite packages.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential libhdf5-dev python3 \
    python3-dev python3-cffi libseccomp-dev luajit libluajit-5.1-dev libsodium-dev meson ninja-build pkg-config \
    git devscripts debhelper

# Clone code
RUN mkdir -p /root/build-deb
RUN git clone ${GIT_URL} /root/build-deb/hdf5-udf

# Do the actual build.
WORKDIR /root/build-deb/hdf5-udf
RUN debuild -us -uc

# Copy artifacts into sandbox container.
FROM alpine:latest
COPY --from=hdf-ud5-builder /root/build-deb /root/hdf5-udf

ENTRYPOINT ["/bin/ash"]
