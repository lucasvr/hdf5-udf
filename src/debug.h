/*
 * HDF5-UDF: User-Defined Functions for HDF5
 *
 * File: debug.h
 *
 * Debugging and benchmarking routines.
 */
#ifndef __debug_h
#define __debug_h

#include <stdio.h>
#include <ctype.h>
#include <sys/time.h>

#define DEBUG

#ifdef DEBUG
class Benchmark { 
    public:
        Benchmark() { start(); }
        ~Benchmark() {}

        void start() {
            tv = 0;
            gettimeofday(&t1, NULL);
        }

        void stop() {
            gettimeofday(&t2, NULL);

            // Adjust tv
            if (t2.tv_sec == t1.tv_sec)
                tv += (1e-6 * t2.tv_usec) - (1e-6 * t1.tv_usec);
            else if (t2.tv_sec > t1.tv_sec) {
                tv += t2.tv_sec - t1.tv_sec;
                if (t2.tv_usec >= t1.tv_usec)
                    tv += (1e-6 * t2.tv_usec - 1e-6 * t1.tv_usec);
                else {
                    tv -= 1;
                    tv += (1 - 1e-6 * t1.tv_usec) + (1e-6 * t2.tv_usec);
                }
            }
        }

        double elapsed() {
            // Elapsed time (seconds)
            stop();
            return tv;
        }

        void print(std::string msg) {
            fprintf(stdout, "%s: %.2f seconds\n", msg.c_str(), elapsed());
        }
    
    private:
        struct timeval t1;
        struct timeval t2;
        double tv;
};
#else
class Benchmark {
    public:
        void print(std::string msg) {}
};
#endif

static inline void asciidump(const char *data, size_t size)
{
    printf("  |");
    for (size_t i=0; i<size; ++i)
        printf("%c", isprint(data[i]) ? data[i] : '.');
    printf("|\n");
}

static inline void hexdump(const char *data, size_t size, size_t maxlines=1000)
{
    for (size_t i=1, lines=0; i<=size; ++i) {
        printf("%02x ", ((char) data[i-1]) & 0xff);
        if (i == size) {
            for (size_t j=0; j<16-(i%16); ++j)
                printf("   ");
            if (i%16 <= 8)
                printf("  ");
            asciidump(&data[i-(i%16)], i%16);
            printf("\n");
        } else if (i%16 == 0) {
            asciidump(&data[i-16], 16);
            if (lines++ == maxlines) {
                printf("...\n");
                break;
            }
        } else if (i%8 == 0) {
            if (i == size)
                asciidump(data, size);
            else
                printf("  ");
        }
    }
}

#endif // __debug_h
