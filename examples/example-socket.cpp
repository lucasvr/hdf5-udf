/*
 * Socket example: retrieves the Linux logo bitmap from the internet
 * using plain HTTP. 
 *
 * To embed this dataset in an existing HDF5 file, run:
 *
 * $ make files
 * $ hdf5-udf example-socket.h5 example-socket.cpp Tux:60x80:uint16
 *
 * To check its output you can use our readh5 utility:
 * $ readh5 example-socket.h5 Tux
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <string>

ssize_t retrieveRemoteFile(char *pbm_data, ssize_t max_size);

extern "C" void dynamic_dataset()
{
    // UDF data and its dimensions
    auto udf_data = lib.getData<uint16_t>("Tux");
    auto udf_dims = lib.getDims("Tux");

    // Retrieve file over plain HTTP
    char *pbm_data = new char[1024*64];
    char *pbm = pbm_data;
    memset(pbm_data, 0, 1024*64);
    ssize_t pbm_size = retrieveRemoteFile(pbm_data, 1024*64);
    if (pbm_size == 0)
    {
        delete[] pbm_data;
        return;
    }

    // Skip the first 3 lines of PBM metadata
    pbm = strchr(pbm, '\n');
    pbm = pbm ? strchr(pbm+1, '\n') : NULL;
    pbm = pbm ? strchr(pbm+1, '\n') : NULL;
    if (! pbm)
    {
        fprintf(stderr, "Invalid PBM file retrieved\n");
        delete[] pbm_data;
        return;
    }
    pbm++;

    // Populate the HDF5 dataset with the PBM data
    ssize_t dims_size = udf_dims[0] * udf_dims[1];
    for (ssize_t i=0, pbm_i=0; pbm_i<pbm_size && i<dims_size; ++pbm_i)
    {
        if (pbm[pbm_i] == '\n')
            continue;
        udf_data[i++] = pbm[pbm_i] == '1' ? 1 : 0;
    }
    delete[] pbm_data;
    return;
}

ssize_t retrieveRemoteFile(char *pbm_data, ssize_t max_size)
{
    // Retrieve the remote file. We know beforehand that it's a PBM file
    // with a resolution of 60x80 pixels
    std::string host = "dotsrc.dl.osdn.net";
    std::string path = "/mirrors/parabola/other/linux-libre/logos/logo_linux_mono.pbm";
    std::string url = "http://" + host + path;

    struct hostent *server = gethostbyname(host.c_str());
    if (! server)
    {
        fprintf(stderr, "Failed to lookup remote host\n");
        return 0;
    }

    struct sockaddr_in remote;
    memset(&remote, 0, sizeof(remote));
    remote.sin_family = AF_INET;
    remote.sin_port = htons(80);
    memcpy(&remote.sin_addr.s_addr, server->h_addr, server->h_length);

    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        fprintf(stderr, "Failed to create socket\n");
        return 0;
    }
    if (connect(sock, (struct sockaddr *) &remote, sizeof(remote)) < 0)
    {
        fprintf(stderr, "Failed to connect to remote host\n");
        close(sock);
        return 0;
    }

    std::string msg = "GET " + path + " HTTP/1.1\r\n" + "Host: " + host + "\r\n\r\n";
    if (send(sock, msg.c_str(), msg.length(), 0) != static_cast<ssize_t>(msg.length()))
    {
        fprintf(stderr, "Error sending GET request\n");
        close(sock);
        return 0;
    }

    ssize_t pbm_i = 0;
    while (max_size >= 0) {
        fd_set read_fds;
        FD_ZERO(&read_fds);
        FD_SET(sock, &read_fds);
        struct timeval tv = { .tv_sec = 1, .tv_usec = 0};
        int ret = select(sock+1, &read_fds, NULL, NULL, &tv);
        if (ret < 0 || (ret == 0 && pbm_i > 0))
            break;

        ssize_t n = read(sock, &pbm_data[pbm_i], max_size);
        if (n <= 0)
            break;
        pbm_i += n;
        max_size -= n;
    }

    if (pbm_i == 0)
        fprintf(stderr, "Failed to retrieve remote file\n");
    else
    {
        // Remove HTTP header
        char *pbm_header = strstr(pbm_data, "P1\n");
        if (! pbm_header)
        {
            fprintf(stderr, "Unexpected HTTP body contents\n");
            pbm_i = 0;
        }
        else
        {
            pbm_i -= (pbm_header-pbm_data);
            memmove(pbm_data, pbm_header, pbm_i);
        }
    }
    close(sock);
    return pbm_i;
}
