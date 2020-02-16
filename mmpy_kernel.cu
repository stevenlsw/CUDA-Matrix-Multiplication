// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include "utils.h"
#include "types.h"

#define BLOCK_SIZE 32 // matrix block
#if BLOCK_SIZE % BLOCKDIM_X || BLOCK_SIZE % BLOCKDIM_Y
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

    __shared__ double As[BLOCK_SIZE][BLOCK_SIZE], Bs[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;

    int I =  by * BLOCK_SIZE+ ty, J =  bx * BLOCK_SIZE + tx;

    _DOUBLE_ c[Y_SUB][X_SUB] = {0}; // Zero initialize the whole array


    for (int kk=0; kk<N/BLOCK_SIZE+1; kk++)
    {
            // load corresponding values of A in the matrix block
        for (int i = 0; i < BLOCK_SIZE; i += BLOCKDIM_Y) {
            for (int j = 0; j < BLOCK_SIZE; j += BLOCKDIM_X) {
                // load I,K of A, As[ty][tx] = A_ELEMENT(I, kk*BLOCK_SIZE + tx);
                As[ty + i][tx + j] = A_ELEMENT(I + ty + i, kk*BLOCK_SIZE + tx + j);
            }
        }
        
        // load corresponding values of B in the matrix block
        for (int i = 0; i < BLOCK_SIZE; i += BLOCKDIM_Y) {
            for (int j = 0; j < BLOCK_SIZE; j += BLOCKDIM_X) {
                // load K,J of B, Bs[ty][tx] = B_ELEMENT(kk*BLOCK_SIZE + ty, J);
                Bs[ty + i][tx + j] = B_ELEMENT(kk*BLOCK_SIZE + ty + i, J + tx + j);
            }
        }

        __syncthreads();

        for (int k=0; k<BLOCK_SIZE; k++)
        {
            for (int i = 0; i < Y_SUB; i++) 
            {
                for (int j = 0; j < X_SUB; j++) 
                {
                    // Compute I,J of C, c += As[ty][k] * Bs[k][tx];
                    c[i][j] += As[ty + i * BLOCKDIM_Y][k] * Bs[k][tx + j * BLOCKDIM_X];
                }
            }
        }

        __syncthreads();

        for (int i = 0; i < Y_SUB; ++i) 
        {
            for (int j = 0; j < X_SUB; ++j) 
            {
                if (I + ty + i * BLOCKDIM_Y < N && J + tx + j * BLOCKDIM_X < N)
                {
                    C_ELEMENT(I + ty + i * BLOCKDIM_Y, J + tx + j * BLOCKDIM_X) = c[i][j];
                }                
            }
        }
    }
}
