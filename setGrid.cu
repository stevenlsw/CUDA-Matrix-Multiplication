#define BLOCK_SIZE_M 96
#define BLOCK_SIZE_N 96

#define ROUND_UP(n, d) (n + d - 1) / d

void setGrid(int n, dim3 &blockDim, dim3 &gridDim) {
    // set your block dimensions and grid dimensions here
    gridDim.x = ROUND_UP(n, BLOCK_SIZE_N);
    gridDim.y = ROUND_UP(n, BLOCK_SIZE_M);
}