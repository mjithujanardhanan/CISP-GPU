
#include<cuda_runtime.h>
#include<stdio.h>
#include<cmath>
#include<stdlib.h>
#include<random>
#include<array>
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include<pybind11/stl.h>

#define block_dim 16                                                                                                                                                                        // Dimension of each cuda block. Each block contains 256 threads representing a pixel each
#define block_size 256    

namespace py=pybind11;


__global__ void DEBAYER_kernel_1(int* Image , int* output, int orientation,int width, int length)                                       //green interpolation
{
    int x=blockIdx.x * block_dim + threadIdx.x, y=blockIdx.y * block_dim + threadIdx.y;                                                 //int j=blockIdx.x * block_dim + threadIdx.x, i=blockIdx.y * block_dim + threadIdx.y;
    int idx= abs(y-=2)* width  + abs(x-=2);

    __shared__ float buffer[20][21];   // in format [y][x]

    int tx=threadIdx.x, ty=threadIdx.y;  // thread x and thread y
    buffer[tx][ty] =0;
    if(y<length && x<width)
    {
        buffer[ty][tx] = Image[idx];
    }
    else if(y<length+2 && x<width+2)
    {
        
        if(length+1-y == 1) y=length-2;
        else y=length-3;

        if(width+1-x == 1) x=width-2;
        else x=width-3;

        idx= abs(y)* width  + abs(x);
        buffer[ty][tx] = Image[idx];

    }

    __syncthreads(); 

    if(tx>1 && tx<18 && ty>1 && ty<18)
    {
        x=blockIdx.x * block_dim + tx-2, y=blockIdx.y * block_dim + ty-2;
        if(y<length && x<width)
        {
            int idx= y* width  + x;
            if(orientation == 0 || orientation == 3)
            {
                if((x+y)&1)
                {
                    output[idx]=buffer[ty][tx];
                }
                else
                {
                    float dv = fabsf(buffer[ty-1][tx] - buffer[ty+1][tx]) + fabsf(2* buffer[ty][tx] -(buffer[ty-2][tx] + buffer[ty+2][tx]));
                    float dh = fabsf(buffer[ty][tx-1] - buffer[ty][tx+1]) + fabsf(2* buffer[ty][tx] -(buffer[ty][tx-2] + buffer[ty][tx+2]));

                    if (dh>dv)
                    {
                        output[idx] = (int)roundf((buffer[ty-1][tx] + buffer[ty+1][tx])*0.5 + (2* buffer[ty][tx] -(buffer[ty-2][tx] + buffer[ty+2][tx]))*0.25f);
                    }
                    
                    else if (dh<dv)
                    {
                        output[idx] = (int)roundf((buffer[ty][tx-1] + buffer[ty][tx+1])*0.5 + (2* buffer[ty][tx] -(buffer[ty][tx-2] + buffer[ty][tx+2]))*0.25f);
                    }
                    else
                    {
                        output[idx] = (int)roundf((buffer[ty-1][tx] + buffer[ty+1][tx] + buffer[ty][tx-1] + buffer[ty][tx+1])*0.25 + (2* buffer[ty][tx] -(buffer[ty-2][tx] + buffer[ty+2][tx]) +2* buffer[ty][tx] -(buffer[ty][tx-2] + buffer[ty][tx+2]))*0.125f);
                    }
                    
                }
            }

            else if(orientation == 1 || orientation == 2)
            {
                if(!((x+y)&1))
                {
                    output[idx]=buffer[ty][tx];
                }
                else
                {
                    float dv = fabsf(buffer[ty-1][tx] - buffer[ty+1][tx]) + fabsf(2* buffer[ty][tx] -(buffer[ty-2][tx] + buffer[ty+2][tx]));
                    float dh = fabsf(buffer[ty][tx-1] - buffer[ty][tx+1]) + fabsf(2* buffer[ty][tx] -(buffer[ty][tx-2] + buffer[ty][tx+2]));

                    if (dh>dv)
                    {
                        output[idx] = (int)roundf((buffer[ty-1][tx] + buffer[ty+1][tx])*0.5 + (2* buffer[ty][tx] -(buffer[ty-2][tx] + buffer[ty+2][tx]))*0.25f);
                    }
                    
                    else if (dh<dv)
                    {
                        output[idx] = (int)roundf((buffer[ty][tx-1] + buffer[ty][tx+1])*0.5 + (2* buffer[ty][tx] -(buffer[ty][tx-2] + buffer[ty][tx+2]))*0.25f);
                    }
                    else
                    {
                        output[idx] = (int)roundf((buffer[ty-1][tx] + buffer[ty+1][tx] + buffer[ty][tx-1] + buffer[ty][tx+1])*0.25 + (2* buffer[ty][tx] -(buffer[ty-2][tx] + buffer[ty+2][tx]) +2* buffer[ty][tx] -(buffer[ty][tx-2] + buffer[ty][tx+2]))*0.125f);
                    }
                    
                }
            }
        }
    }
}


// kernel for debayering the input image 
__global__ void DEBAYER_kernel_2(int* Image , int* green, int* red, int* blue, int orientation,int width, int length)                   //color interpolation
{
    int x=blockIdx.x * block_dim + threadIdx.x, y=blockIdx.y * block_dim + threadIdx.y;                                                 //int j=blockIdx.x * block_dim + threadIdx.x, i=blockIdx.y * block_dim + threadIdx.y;
    int idx= abs(y-=2)* width  + abs(x-=2);

    __shared__ float buffer1[20][21], bufferg[20][21] ;

    int tx=threadIdx.x, ty=threadIdx.y;

    buffer1[ty][tx]=0;
    bufferg[ty][tx]=0;

    if(y<length && x<width)
    {
        buffer1[ty][tx] = Image[idx];
        bufferg[ty][tx] = green[idx];
    }
    else if(y<length+2 && x<width+2)
    {
        
        if(length+1-y == 1) y=length-2;
        else y=length-3;

        if(width+1-x == 1) x=width-2;
        else x=width-3;

        idx= abs(y)* width  + abs(x);
        buffer1[ty][tx] = Image[idx];
        bufferg[ty][tx] = green[idx];

    }

    __syncthreads(); 

    if(tx>1 && tx<18 && ty>1 && ty<18)
    {
        int x=blockIdx.x * block_dim + tx-2, y=blockIdx.y * block_dim + ty-2;
        if(y<length && x<width)
        {
            int idx= y* width  + x;
            if(orientation == 0 || orientation == 3)
            {
                if((x+y)&1)
                {
                    if(orientation ==0)
                    {
                        if(x&1 && !(y&1))
                        {
                            red[idx] = (int)roundf(bufferg[ty][tx]+0.5f *(buffer1[ty-1][tx]-bufferg[ty-1][tx] + (buffer1[ty+1][tx]-bufferg[ty+1][tx])));                     //vertical interpolation
                            blue[idx]= (int)roundf(bufferg[ty][tx]+0.5f *(buffer1[ty][tx-1]-bufferg[ty][tx-1] + (buffer1[ty][tx+1]-bufferg[ty][tx+1])));                     //horizontal interpolation
                        }
                        else
                        {
                            blue[idx] = (int)roundf(bufferg[ty][tx]+0.5f *(buffer1[ty-1][tx]-bufferg[ty-1][tx] + (buffer1[ty+1][tx]-bufferg[ty+1][tx])));                    //vertical interpolation
                            red[idx]  = (int)roundf(bufferg[ty][tx]+0.5f *(buffer1[ty][tx-1]-bufferg[ty][tx-1] + (buffer1[ty][tx+1]-bufferg[ty][tx+1])));                    //horizontal interpolation
                        }
                    }
                    else
                    {
                        if(x&1 && !(y&1))
                        {
                            blue[idx] = (int)roundf(bufferg[ty][tx]+0.5f *(buffer1[ty-1][tx]-bufferg[ty-1][tx] + (buffer1[ty+1][tx]-bufferg[ty+1][tx])));                    //vertical interpolation
                            red[idx]  = (int)roundf(bufferg[ty][tx]+0.5f *(buffer1[ty][tx-1]-bufferg[ty][tx-1] + (buffer1[ty][tx+1]-bufferg[ty][tx+1])));                    //horizontal interpolation
                            
                        }
                        else
                        {
                            red[idx] = (int)roundf(bufferg[ty][tx]+0.5f *(buffer1[ty-1][tx]-bufferg[ty-1][tx] + (buffer1[ty+1][tx]-bufferg[ty+1][tx])));                     //vertical interpolation
                            blue[idx]= (int)roundf(bufferg[ty][tx]+0.5f *(buffer1[ty][tx-1]-bufferg[ty][tx-1] + (buffer1[ty][tx+1]-bufferg[ty][tx+1])));                     //horizontal interpolation
                        }
                    }
                }
                else
                {
                    if(orientation ==0)
                    {
                        if(y&1 && x&1)
                        {
                            red[idx] = (int)roundf(buffer1[ty][tx]);
                            blue[idx] =(int)roundf(bufferg[ty][tx]+0.25f*(buffer1[ty-1][tx-1]-bufferg[ty-1][tx-1] + buffer1[ty+1][tx+1]-bufferg[ty+1][tx+1] +buffer1[ty-1][tx+1]-bufferg[ty-1][tx+1] + buffer1[ty+1][tx-1]-bufferg[ty+1][tx-1]));
                        }
                        else 
                        {
                            blue[idx] = (int)roundf(buffer1[ty][tx]);
                            red[idx]  = (int)roundf(bufferg[ty][tx]+0.25f*(buffer1[ty-1][tx-1]-bufferg[ty-1][tx-1] + buffer1[ty+1][tx+1]-bufferg[ty+1][tx+1] +buffer1[ty-1][tx+1]-bufferg[ty-1][tx+1] + buffer1[ty+1][tx-1]-bufferg[ty+1][tx-1]));
                        }
            
                    }
                    else
                    {
                        if(y&1 && x&1)
                        {
                            blue[idx] = (int)roundf(buffer1[ty][tx]);
                            red[idx]  = (int)roundf(bufferg[ty][tx]+0.25f*(buffer1[ty-1][tx-1]-bufferg[ty-1][tx-1] + buffer1[ty+1][tx+1]-bufferg[ty+1][tx+1] +buffer1[ty-1][tx+1]-bufferg[ty-1][tx+1] + buffer1[ty+1][tx-1]-bufferg[ty+1][tx-1]));
                        }
                        else 
                        {
                            red[idx]  = (int)roundf(buffer1[ty][tx]);
                            blue[idx] = (int)roundf(bufferg[ty][tx]+0.25f*(buffer1[ty-1][tx-1]-bufferg[ty-1][tx-1] + buffer1[ty+1][tx+1]-bufferg[ty+1][tx+1] +buffer1[ty-1][tx+1]-bufferg[ty-1][tx+1] + buffer1[ty+1][tx-1]-bufferg[ty+1][tx-1]));
                        }


                    }
                    
                }
            }

            if(orientation == 1 || orientation == 2)
            {
                if(!((y+x)&1))
                {
                    if(orientation ==1)
                    {
                        if(x&1 && y&1)
                        {
                            blue[idx] = (int)roundf(bufferg[ty][tx]+0.5f *(buffer1[ty-1][tx]-bufferg[ty-1][tx] + (buffer1[ty+1][tx]-bufferg[ty+1][tx])));                    //vertical interpolation
                            red[idx]  = (int)roundf(bufferg[ty][tx]+0.5f *(buffer1[ty][tx-1]-bufferg[ty][tx-1] + (buffer1[ty][tx+1]-bufferg[ty][tx+1])));                    //horizontal interpolation
                            
                        }
                        else
                        {
                            red[idx] = (int)roundf(bufferg[ty][tx]+0.5f *(buffer1[ty-1][tx]-bufferg[ty-1][tx] + (buffer1[ty+1][tx]-bufferg[ty+1][tx])));                     //vertical interpolation
                            blue[idx]= (int)roundf(bufferg[ty][tx]+0.5f *(buffer1[ty][tx-1]-bufferg[ty][tx-1] + (buffer1[ty][tx+1]-bufferg[ty][tx+1])));                     //horizontal interpolation
                        }
                    }
                    else
                    {
                        if(x&1 && y&1)
                        {
                            red[idx] = (int)roundf(bufferg[ty][tx]+0.5f *(buffer1[ty-1][tx]-bufferg[ty-1][tx] + (buffer1[ty+1][tx]-bufferg[ty+1][tx])));                     //vertical interpolation
                            blue[idx]= (int)roundf(bufferg[ty][tx]+0.5f *(buffer1[ty][tx-1]-bufferg[ty][tx-1] + (buffer1[ty][tx+1]-bufferg[ty][tx+1])));                     //horizontal interpolation
                        }
                        else
                        {
                            blue[idx] = (int)roundf(bufferg[ty][tx]+0.5f *(buffer1[ty-1][tx]-bufferg[ty-1][tx] + (buffer1[ty+1][tx]-bufferg[ty+1][tx])));                    //vertical interpolation
                            red[idx]  = (int)roundf(bufferg[ty][tx]+0.5f *(buffer1[ty][tx-1]-bufferg[ty][tx-1] + (buffer1[ty][tx+1]-bufferg[ty][tx+1])));                    //horizontal interpolation
                        }
                    }
                }
                else
                {
                    if(orientation ==1)
                    {
                        if(!(y&1) && x&1)
                        {
                            blue[idx] =(int)roundf(buffer1[ty][tx]);
                            red[idx]  = (int)roundf(bufferg[ty][tx]+0.25f*(buffer1[ty-1][tx-1]-bufferg[ty-1][tx-1] + buffer1[ty+1][tx+1]-bufferg[ty+1][tx+1] +buffer1[ty-1][tx+1]-bufferg[ty-1][tx+1] + buffer1[ty+1][tx-1]-bufferg[ty+1][tx-1]));
                        }
                        else 
                        {
                            red[idx]  = (int)roundf(buffer1[ty][tx]);
                            blue[idx] = (int)roundf(bufferg[ty][tx]+0.25f*(buffer1[ty-1][tx-1]-bufferg[ty-1][tx-1] + buffer1[ty+1][tx+1]-bufferg[ty+1][tx+1] +buffer1[ty-1][tx+1]-bufferg[ty-1][tx+1] + buffer1[ty+1][tx-1]-bufferg[ty+1][tx-1]));
                        }
            
                    }
                    else
                    {
                        if(!(y&1) && x&1)
                        {
                            red[idx]  = (int)roundf(buffer1[ty][tx]);
                            blue[idx] = (int)roundf(bufferg[ty][tx]+0.25f*(buffer1[ty-1][tx-1]-bufferg[ty-1][tx-1] + buffer1[ty+1][tx+1]-bufferg[ty+1][tx+1] +buffer1[ty-1][tx+1]-bufferg[ty-1][tx+1] + buffer1[ty+1][tx-1]-bufferg[ty+1][tx-1]));
                        } 
                        else 
                        {
                            blue[idx] = (int)roundf(buffer1[ty][tx]);
                            red[idx]  = (int)roundf(bufferg[ty][tx]+0.25f*(buffer1[ty-1][tx-1]-bufferg[ty-1][tx-1] + buffer1[ty+1][tx+1]-bufferg[ty+1][tx+1] +buffer1[ty-1][tx+1]-bufferg[ty-1][tx+1] + buffer1[ty+1][tx-1]-bufferg[ty+1][tx-1]));
                        }


                    }
                    
                }
            }
        }
    }
}




py::tuple Bayer(py::array_t<int> Image, int orientation, int width, int length)   // BGGR - 0,  GBRG -1, GRBG -2, RGGB -3 
{
    auto buffer = Image.request();
    int *Input = static_cast<int*>(buffer.ptr);
    int array_size = static_cast<int>(buffer.size);

    int *D_Image_1;                                                                                                                                                             //pointer declaration for gpu memory creation.

    cudaMalloc( &D_Image_1, array_size * sizeof(int));                                                                                                                                      // creating memory pointers on gpu memory for image.
    cudaMemcpy(D_Image_1 , Input, array_size * sizeof(int), cudaMemcpyHostToDevice);

    const int blockx= (width%16 == 0)?(width/16):(width/16 +1),blocky= (length%16 == 0)?(length/16):(length/16 +1);

    int *D_Image_Gr, *D_Image_Rd, *D_Image_Bl;  // array for storing rgb after debayering

    cudaMalloc( &D_Image_Gr, array_size * sizeof(int));
    cudaMalloc( &D_Image_Rd, array_size * sizeof(int));
    cudaMalloc( &D_Image_Bl, array_size * sizeof(int));

    DEBAYER_kernel_1<<<dim3(blockx,blocky),dim3(20,20)>>>(D_Image_1, D_Image_Gr, orientation, width, length);
    cudaDeviceSynchronize();

    DEBAYER_kernel_2<<<dim3(blockx,blocky),dim3(20,20)>>>(D_Image_1, D_Image_Gr ,D_Image_Rd, D_Image_Bl, orientation, width, length);


    py::array_t<int> Red({length, width});
    py::array_t<int> Green({length, width});
    py::array_t<int> Blue({length, width});

    auto r_buf = Red.request();
    auto g_buf = Green.request();
    auto b_buf = Blue.request();

    int* R = static_cast<int*>(r_buf.ptr);
    int* G = static_cast<int*>(g_buf.ptr);
    int* B = static_cast<int*>(b_buf.ptr);

    cudaMemcpy(R, D_Image_Rd, array_size*sizeof(int), cudaMemcpyDeviceToHost);

    cudaMemcpy(G, D_Image_Gr, array_size*sizeof(int), cudaMemcpyDeviceToHost);

    cudaMemcpy(B, D_Image_Bl, array_size*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(D_Image_1);
    cudaFree(D_Image_Gr);
    cudaFree(D_Image_Rd);
    cudaFree(D_Image_Bl);


    return py::make_tuple(Red, Green, Blue);



}

PYBIND11_MODULE(bayer, m)
{
    m.def("Bayer", &Bayer, "perform Debayer on an image");
}