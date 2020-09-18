#pragma once

#include "allocation.hpp"
#include <stdio.h>
float** alloc_2d_init(int r, int c)
{
    float** A = new float*[r];
    A[0] = new  float[r*c];
    for (int i = 1; i < r; ++i)
        A[i] = A[i-1] + c;
    return A;
}

void data_init(float** A, int r, int c)
{
    for(int i=0; i<r; i++)
    {
        for(int j=0; j<c; j++)
        {
            A[i][j]=(float)(i*c+j);
        }
    }
}

void print_helper(float* A, int r, int c)
{

        for(int i =0; i<r*c; i++)
        {
            printf("%f, ", A[i]);
        }
    printf("\n");
}