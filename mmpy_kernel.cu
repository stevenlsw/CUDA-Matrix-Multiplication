// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include "utils.h"
#include "types.h"

#define BLOCK_SIZE 32 # BLOCK_SIZE=BLOCK_DIM_X=BLOCK_DIM_Y

using namespace std;

__global__ void matMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {

    __shared__ double As[BLOCK_SIZE][BLOCK_SIZE], Bs[BLOCK_SIZE][BLOCK_SIZE];

    int I =  blockIdx.y*BLOCK_SIZE + threadIdx.y;
    int J =  blockIdx.x*BLOCK_SIZE + threadIdx.x;

    _DOUBLE_ _c = 0;

    if((I < N) && (J < N))
    {
        for (int kk=0; kk<N/BLOCK_SIZE; kk++)
        {
            As[threadIdx.y][threadIdx.x] = A[I*N + kk*BLOCK_SIZE + threadIdx.x];
            Bs[threadIdx.y][threadIdx.x] = B[(kk*BLOCK_SIZE+threadIdx.y) * N + J];
            __syncthreads();
            for (int k=0; k<BLOCK_SIZE; k++)
            {
                _c += As[threadIdx.y][k] * Bs[k][threadIdx.x]
            }
            __syncthreads();
        }
        C[I * N + J] = _c;
    }
}
