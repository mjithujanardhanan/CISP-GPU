#include<cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include<conio.h>

#define N 10000000
#define block_size 256
#define W_1 2
#define L_1 3
#define W_2 3
#define L_2 2



void init_vector(double *vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = (double)rand() ;
    }
}


__global__ void vector_add(double* a, double* b, long n)
{
    long idx= blockIdx.x * blockDim.x+ threadIdx.x;

    if(idx<n)
    {
        a[idx] +=b[idx];
        //printf("%d",idx);
    }
}


__global__ void mat_mul(double* a, double* b, double* c, long n)
{
    long idx= blockIdx.x * blockDim.x+ threadIdx.x;

    if(idx < n){
        int i=idx/W_1,j=idx%W_1;
        double sum=0;
        int count=0;

        while(count<W_2)                                                                                                                                                                                                         
        {
            sum+= a[i*L_1 + count] * b[count*L_2 + j];
            count++;
        }
        c[idx] = sum;
   
    }

}



int main()
{


    if(L_1 != W_2) return 0;

    long count1 = W_1 *L_1;
    long count2 = W_2 *L_2; 
    long count3 = W_1 *L_2 ; 



    double *h_a , *h_b, *h_c;
    double *d_a , *d_b, *d_c;

    size_t size1 = count1 * sizeof(double);
    size_t size2 = count2 * sizeof(double);
    size_t size3 = count3 * sizeof(double);

    h_a = (double*)malloc(size1);
    h_b = (double*)malloc(size2);
    h_c = (double*)malloc(size3);
    


    init_vector(h_a, count1);
    init_vector(h_b, count2);

    const int grid_size = (count3 %block_size != 0 )?(count3 /block_size +1):(count3 /block_size);

    cudaMalloc(&d_a, size1);
    cudaMalloc(&d_b, size2);
    cudaMalloc(&d_c, size3);

    cudaMemcpy(d_a,h_a,size1,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,size2,cudaMemcpyHostToDevice);


    mat_mul<<<grid_size,block_size>>>(d_a,d_b,d_c, count3); 

    cudaDeviceSynchronize();

    
    cudaMemcpy(h_c,d_c,size3,cudaMemcpyDeviceToHost);
    
    for(int i=0;i<count1;i++)
    {
        printf("%f \t", h_a[i]);
        if((i+1) % L_1 == 0)
            printf("\n");
    }
    printf("\n");
    for(int i=0;i<count2;i++)
    {
        printf("%f \t", h_b[i]);
        if((i+1) % L_2 == 0)
            printf("\n");
    }
    printf("\n");
    for(int i=0;i<count3;i++)
    {
        printf("%f \t", h_c[i]);
        if((i+1) % W_1 == 0)
            printf("\n");
    }
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    getch();
    return 0;
}