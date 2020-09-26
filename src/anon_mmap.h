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
#include <sys/mman.h>
#include <errno.h>

class AnonymousMemoryMap {
public:
    AnonymousMemoryMap(size_t size) :
        mm_size(size)
    {
    }

    ~AnonymousMemoryMap()
    {
        munmap(mm, mm_size);
    }

    bool create()
    {
        mm = mmap(NULL, mm_size, PROT_READ|PROT_WRITE, MAP_SHARED|MAP_ANONYMOUS, -1, 0);
        if (mm == (void *) -1)
            fprintf(stderr, "Failed to create anonymous mapping: %s\n", strerror(errno));
        return mm != (void *) -1;
    }

    void *mm;
    size_t mm_size;
};

#endif /* __anon_mmap_h */