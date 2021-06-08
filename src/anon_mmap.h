/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: anon_mmap.h
 *
 * Helper class to manage the life and death of a memory map.
 */
#ifndef __anon_mmap_h
#define __anon_mmap_h

#include <stdio.h>
#include <errno.h>

#ifndef __MINGW64__
#include <sys/mman.h>
#endif

class AnonymousMemoryMap {
public:
    AnonymousMemoryMap(size_t size) :
        mm_size(size)
    {
    }

    ~AnonymousMemoryMap()
    {
#ifndef __MINGW64__
        munmap(mm, mm_size);
#endif
    }

    bool createMapFor(void *output_dataset)
    {
#ifdef __MINGW64__
        // On Windows, this class simply serves as a container that holds a pointer
        // to the output dataset (i.e., void *mm). That will change once support for
        // sandbox is extended to that OS.
        mm = output_dataset;
#else
        (void) output_dataset;
        mm = mmap(NULL, mm_size, PROT_READ|PROT_WRITE, MAP_SHARED|MAP_ANONYMOUS, -1, 0);
        if (mm == (void *) -1)
            fprintf(stderr, "Failed to create anonymous mapping: %s\n", strerror(errno));
#endif
        return mm != (void *) -1;
    }

    void *mm;
    size_t mm_size;
};

#endif /* __anon_mmap_h */
