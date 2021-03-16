/*
 * Play Doom through a user-defined HDF5 dataset!
 * 
 * This example relies on a modified Doom engine that shares the display
 * over a System V shared memory segment. You can download it from
 * https://github.com/lucasvr/chocolate-doom.
 * 
 * Each time the dataset is read we attach to the shared memory segment
 * and retrieve the framebuffer and the palette. We then encode that data
 * as a regular 2D dataset and let h5dump and our own readh5 tools render
 * it nicely.
 *
 * To embed this dataset in an existing HDF5 file, run:
 *
 * $ make files
 * $ hdf5-udf example-doom.h5 example-doom.cpp Framebuffer:320x200:uint8 Palette:7680:uint8
 *
 * Note that the Palette size is 10x bigger than it needs to be. Once issue
 * https://github.com/lucasvr/hdf5-udf/issues/7 is resolved we can make it
 * into an array of 768 elements.
 *
 * The output looks best when rendered by readh5 in a loop:
 * $ while true; do readh5 example-doom.h5 Framebuffer Palette; done
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
#include <sys/shm.h>
#include <sys/ipc.h>
#include <string>

typedef struct { uint8_t r; uint8_t g; uint8_t b; } pal_t;
bool retrieveDisplay(uint8_t *display, ssize_t width, ssize_t height, pal_t *pal, ssize_t pal_size);

extern "C" void dynamic_dataset()
{
    // We export two datasets. The first is where the video dump
    // is captured ("Framebuffer"). The second holds the RGB values
    // to be used when rendering the framebuffer ("Palette").
    auto fb_data  = lib.getData<uint8_t>("Framebuffer");
    auto fb_dims  = lib.getDims("Framebuffer");
    auto pal      = lib.getData<uint8_t>("Palette");
    auto pal_dims = lib.getDims("Palette");

    memset(pal, 0, pal_dims[0]);
    pal_t *ptr = (pal_t *) pal;
    retrieveDisplay(fb_data, fb_dims[0], fb_dims[1], ptr, pal_dims[0]/10);
}

bool retrieveDisplay(uint8_t *display, ssize_t width, ssize_t height, pal_t *pal, ssize_t pal_size)
{
    ssize_t size = (3 * 256) + (width * height);
    int shmid = shmget(666, size, IPC_CREAT|0644);
    if (shmid < 0)
    {
        fprintf(stderr, "Failed to allocate shared memory segment: %s\n", strerror(errno));
        return false;
    }

    void *mm = shmat(shmid, NULL, 0);
    if (mm == (void *) -1)
    {
        fprintf(stderr, "Failed to attach to shared memory segment\n");
        return false;
    }

    memcpy(pal, mm, pal_size);
    memcpy(display, ((char *) mm) + pal_size, width*height);

    shmdt(mm);
    return true;
}