# Fedora Package

The Dockerfile located in this folder can be used to build Fedora package artifacts for this repository.
From the top-level directory, run:

1. `docker build -t hdf5-udf-builder-fedora -f fedora/Dockerfile`
2. `docker run -it hdf5-udf-builder-fedora`
3. `cd /root/rpmbuild`

The `RPMS` and `SRPMS` directories will contain all build artifacts.

4. Use `docker cp` to copy the desired artifacts out of the container.
