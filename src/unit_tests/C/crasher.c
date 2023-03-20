#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/mman.h>
#if !defined(__APPLE__)
#include <malloc.h>
#endif

#define handle_error(msg) \
    do { perror(msg); exit(EXIT_FAILURE); } while (0)

char *buffer;
int flag=0;

int main(int argc, char *argv[])
{
    char *p; char a;
    int pagesize;

    pagesize=4096;

    /* Allocate a buffer aligned on a page boundary;
       initial protection is PROT_READ | PROT_WRITE */

#ifdef __APPLE__
    buffer = valloc(4 * pagesize);
#else
    buffer = memalign(pagesize, 4 * pagesize);
#endif
    if (buffer == NULL)
        handle_error("memalign");

    printf("Start of region:        0x%lx\n", (long) buffer);
    printf("Start of region:        0x%lx\n", (long) buffer+pagesize);
    printf("Start of region:        0x%lx\n", (long) buffer+2*pagesize);
    printf("Start of region:        0x%lx\n", (long) buffer+3*pagesize);
    //if (mprotect(buffer + pagesize * 0, pagesize,PROT_NONE) == -1)
    if (mprotect(buffer + pagesize * 0, pagesize,PROT_NONE) == -1)
        handle_error("mprotect");

    //for (p = buffer ; ; )
    if(flag==0)
    {
        p = buffer+pagesize/2;
        printf("It comes here before reading memory\n");
        a = *p; //trying to read the memory
        printf("It comes here after reading memory\n");
    }
    else
    {
        if (mprotect(buffer + pagesize * 0, pagesize,PROT_READ) == -1)
        handle_error("mprotect");
        a = *p;
        printf("Now i can read the memory\n");

    }
/*  for (p = buffer;p<=buffer+4*pagesize ;p++ )
    {
        //a = *(p);
        *(p) = 'a';
        printf("Writing at address %p\n",p);

    }*/

    printf("Loop completed\n");     /* Should never happen */
    exit(EXIT_SUCCESS);
}