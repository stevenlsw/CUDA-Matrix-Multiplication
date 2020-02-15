// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include "utils.h"
#include "types.h"

#define BLOCK_SIZE 32 // BLOCK_SIZE=BLOCK_DIM_X=BLOCK_DIM_Y

#define MAT_ELEMENT(mat, N, i, j) ((i)<N && (j)<N ? mat[(i)*N+(j)] : 0)
#define A_ELEMENT(i, j) MAT_ELEMENT(A, N, i, j)
#define B_ELEMENT(i, j) MAT_ELEMENT(B, N, i, j)

using namespace std;

__global__ void matMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {

    __shared__ double As[BLOCK_SIZE][BLOCK_SIZE], Bs[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;

    int I =  by * BLOCK_SIZE+ ty, J =  bx * BLOCK_SIZE + tx;

    _DOUBLE_ c = 0;

    if((I < N) && (J < N))
    {
        for (int kk=0; kk<N/BLOCK_SIZE+1; kk++)
        {
            // load I,K of A
            As[ty][tx] = A_ELEMENT(I, kk*BLOCK_SIZE + tx);
            // load K,J of B
            Bs[ty][tx] = B_ELEMENT(kk*BLOCK_SIZE + ty, J);

            __syncthreads();

            // Compute I,J of C
            for (int k=0; k<BLOCK_SIZE; k++)
            {
                c += As[ty][k] * Bs[k][tx];
            }
            __syncthreads();
            C[I * N + J] = c;
        }
    }
}
