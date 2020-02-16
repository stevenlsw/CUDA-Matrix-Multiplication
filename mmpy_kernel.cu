// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include "utils.h"
#include "types.h"

#define BLOCK_SIZE 64 // matrix block
#define BLOCK_SIZE_K 32 // sub block 

#if BLOCK_SIZE % BLOCKDIM_X || BLOCK_SIZE % BLOCKDIM_Y || \
    BLOCK_SIZE_K % BLOCKDIM_X || BLOCK_SIZE_K % BLOCKDIM_Y
#error BLOCK_SIZE must be multiple of blockDim
#endif

// Number of instructions Computing in each thread
#define X_SUB (BLOCK_SIZE / BLOCKDIM_X)
#define Y_SUB (BLOCK_SIZE / BLOCKDIM_Y)

#define MAT(mat, N, i, j) (mat[(i)*N + (j)])
#define MAT_PADDED(mat, N, i, j) ((i) < N && (j) < N ? MAT(mat, N, i, j) : 0)
#define A_ELEMENT(i, j) MAT_PADDED(A, N, i, j)
#define B_ELEMENT(i, j) MAT_PADDED(B, N, i, j)
#define C_ELEMENT(i, j) MAT(C, N, i, j)

using namespace std;

__global__ void matMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) 
{

    __shared__ double As[BLOCK_SIZE][BLOCK_SIZE_K], Bs[BLOCK_SIZE_K][BLOCK_SIZE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;

    int I0 =  by * BLOCK_SIZE, J0 =  bx * BLOCK_SIZE;

    _DOUBLE_ c[Y_SUB][X_SUB] = {0}; // Zero initialize the whole array

    for (int kk=0; kk<N/BLOCK_SIZE_K+1; kk++)
    {
            // load corresponding values of A in the matrix block
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; i += BLOCKDIM_Y) {
            #pragma unroll
            for (int j = 0; j < BLOCK_SIZE_K; j += BLOCKDIM_X) {
                // load I,K of A, As[ty][tx] = A_ELEMENT(I, kk*BLOCK_SIZE + tx);
                As[ty + i][tx + j] = A_ELEMENT(I0 + ty + i, kk*BLOCK_SIZE_K + tx + j);
            }
        }
        
        // load corresponding values of B in the matrix block
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE_K; i += BLOCKDIM_Y) {
            #pragma unroll
            for (int j = 0; j < BLOCK_SIZE; j += BLOCKDIM_X) {
                // load K,J of B, Bs[ty][tx] = B_ELEMENT(kk*BLOCK_SIZE + ty, J);
                Bs[ty + i][tx + j] = B_ELEMENT(kk*BLOCK_SIZE_K + ty + i, J0 + tx + j);
            }
        }

        __syncthreads();

        for (int k=0; k<BLOCK_SIZE_K; k++)
        {
            #pragma unroll
            for (int i = 0; i < Y_SUB; i++) 
            {
                #pragma unroll
                for (int j = 0; j < X_SUB; j++) 
                {
                    // Compute I,J of C, c += As[ty][k] * Bs[k][tx];
                    c[i][j] += As[ty + i * BLOCKDIM_Y][k] * Bs[k][tx + j * BLOCKDIM_X];
                }
            }
        }

        __syncthreads();
        #pragma unroll
        for (int i = 0; i < Y_SUB; ++i) 
        {
            #pragma unroll
            for (int j = 0; j < X_SUB; ++j) 
            {
                if (I0 + ty + i * BLOCKDIM_Y < N && J0 + tx + j * BLOCKDIM_X < N)
                {
                    C_ELEMENT(I0 + ty + i * BLOCKDIM_Y, J0 + tx + j * BLOCKDIM_X) = c[i][j];
                }                
            }
        }
    }
}
