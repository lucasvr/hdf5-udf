# Debian Package

The Dockerfile located in this folder can be used to build Debian package artifacts for this repository.

1. `cd ./debian-build`
2. `docker build . -t hdf5-udf-builder`
3. `docker run -it hdf5-udf-builder`
4. `cd /root`

The `syscall_intercept` and `hdf5-udf` directories will contain all build artifacts. Any sub-directories were the resources used to build those artifacts.

5. Use `docker cp` to copy the desired artifacts out of the container.
