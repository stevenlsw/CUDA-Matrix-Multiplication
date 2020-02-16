#define BLOCK_SIZE 64

void setGrid(int n, dim3 &blockDim, dim3 &gridDim)
{
   // set your block dimensions and grid dimensions here
   gridDim.x = n / BLOCK_SIZE;
   gridDim.y = n / BLOCK_SIZE;
   if(n % blockDim.x != 0)
   	gridDim.x++;
   if(n % blockDim.y != 0)
    	gridDim.y++;
}
