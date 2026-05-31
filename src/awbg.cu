#include<cuda_runtime.h>
#include<stdio.h>
#include<cmath>
#include<stdlib.h>
#include<random>
#include<conio.h>
#include <algorithm>

#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>

namespace py=pybind11;

#define block_dim 16
#define block_size 256     

/* this program is an calculation of automatic white balance gain on digital bayer domain images*/
__global__ void AWBG_kernel(int* Image , unsigned long long* awbg,int orientation,int width, int length)                                                                                    // || BGGR - 0 ||  GBRG -1 || GRBG -2 || RGGB -3 ||             || awbg- || 0-> green || 1-> blue || 2-> red  ||
{
    int x=blockIdx.x * blockDim.x + threadIdx.x, y=blockIdx.y * blockDim.y + threadIdx.y;                                                                                                   // calculating the pixel coordinates i and j corresponding to the thread.
    long long idx= (y)* width  + (x);                                                                                                                                                       // Idx is the index of the pixel in the flattened array. 

    __shared__ unsigned long long Green_sum[block_size], Red_sum[block_size], Blue_sum[block_size];                                                                                      // Shared memory initialization for each channel   int j=blockIdx.x * blockDim.x + threadIdx.x, i=blockIdx.y * blockDim.y + threadIdx.y;  

    int threadid = threadIdx.y*blockDim.x + threadIdx.x;

    Green_sum[threadid]  = 0; 
    Red_sum[threadid]  = 0; 
    Blue_sum[threadid]  = 0; 

    if(y<length && x<width)
    {
        if(orientation == 0 || orientation == 3)
        {
            if( (x+y)&1)
            {
                Green_sum[threadid] = Image[idx];
            }
            else if(y&1 && x&1)
            {
                if(orientation==0)
                    Red_sum[threadid] = Image[idx];
                else
                    Blue_sum[threadid] = Image[idx];
            }
            else
            {
                if(orientation==0)
                    Blue_sum[threadid] = Image[idx];
                else
                    Red_sum[threadid] = Image[idx];
            }
        }
        else
        {
            if( !((x+y)&1))
            {
                Green_sum[threadid] = Image[idx];
            }
            else if(y&1 && !(x&1))
            {
                if(orientation==1)
                    Red_sum[threadid] = Image[idx];
                else
                    Blue_sum[threadid] = Image[idx];
            }
            else
            {
                if(orientation==1)
                    Blue_sum[threadid] = Image[idx];
                else
                    Red_sum[threadid] = Image[idx];
            }
        }
    }
    __syncthreads();
    

    for(int stride = block_size/2; stride > 0; stride /= 2)
    {
        if(threadid < stride)
        {
            Green_sum[threadid] += Green_sum[threadid + stride];
            Red_sum[threadid] += Red_sum[threadid + stride];
            Blue_sum[threadid] += Blue_sum[threadid + stride];
        }   

        __syncthreads();
    }
        

    if(threadid == 0 )
    {
        
        atomicAdd(&awbg[0],Green_sum[0]);
        atomicAdd(&awbg[1],Blue_sum[0]);
        atomicAdd(&awbg[2],Red_sum[0]);

    }
}

__global__ void AWBG_Apply_kernel(int* Image ,float gain_r,float gain_g,float gain_b,int orientation,int width, int length)                                                                 // || BGGR - 0 ||  GBRG -1 || GRBG -2 || RGGB -3 ||             || awbg- || 0-> green || 1-> blue || 2-> red  ||
{
    int j=blockIdx.x * blockDim.x + threadIdx.x, i=blockIdx.y * blockDim.y + threadIdx.y;
    long long idx= (i)* width  + (j);


    if(i<length && j<width)
    {
        if(orientation == 0 || orientation == 3)
        {
            if(i&1 && j&1)
            {
                if(orientation ==0)
                    Image[idx] = min(65535, int(Image[idx] * gain_r));
                else
                    Image[idx] = min(65535, int(Image[idx] * gain_b));
            }
            else if(!(i&1)&&!(j&1))
            {
                if(orientation ==0)
                    Image[idx] = min(65535, int(Image[idx] * gain_b));
                else
                    Image[idx] = min(65535, int(Image[idx] * gain_r));
            }
        }
        else
        {
            if((i&1) && !(j&1))
            {
                if(orientation ==2)
                    Image[idx] = min(65535, int(Image[idx] * gain_r));
                else
                    Image[idx] = min(65535, int(Image[idx] * gain_b));  
            }
            else if(!(i&1) && (j&1))
            {
                if(orientation ==2)
                    Image[idx] = min(65535, int(Image[idx] * gain_b));
                else
                    Image[idx] = min(65535, int(Image[idx] * gain_r));  
            }
        }
    }

}

py::array_t<int> AWBG(py::array_t<int> Image, int width, int length, int orientation)  //k is the multiplication factor for gain
{
    auto buffer = Image.request();
    if(buffer.ndim != 1)    //error check to see if flattened image is passed.
        throw std::runtime_error("image must be Flattened :: LSC module");
    if(buffer.size < (width * length))
        throw std::runtime_error("Wrong image size :: LSC module");
    int *Input = static_cast<int*>(buffer.ptr);
    int array_size = static_cast<int>(buffer.size);
    unsigned long long* d_awbg;

    cudaMalloc( &d_awbg, 3 * sizeof(unsigned long long));

    cudaMemset( d_awbg, 0, 3 * sizeof(unsigned long long));

    int *D_Image; //pointer declaration for gpu memory creation.
    cudaMalloc( &D_Image, array_size * sizeof(int));// creating memory pointers on gpu memory for image.

    cudaMemcpy(D_Image , Input, array_size * sizeof(int), cudaMemcpyHostToDevice);  //copying image data to gpu memory
    const int blockx= (width%16 == 0)?(width/16):(width/16 +1),blocky= (length%16 == 0)?(length/16):(length/16 +1);


    AWBG_kernel<<<dim3(blockx,blocky),dim3(16,16)>>>(D_Image , d_awbg, orientation, width, length);
    cudaDeviceSynchronize();     
    unsigned long long h_awbg[3];

    cudaMemcpy(h_awbg, d_awbg, 3 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    h_awbg[0] = h_awbg[0]*2/(length *width);
    h_awbg[1] = h_awbg[1]*4/(length *width);
    h_awbg[2] = h_awbg[2]*4/(length *width);
    
    float gain_r = float(h_awbg[0]) / float(h_awbg[2]);
    float gain_g = 1.0f;
    float gain_b = float(h_awbg[0]) / float(h_awbg[1]);

    AWBG_Apply_kernel<<<dim3(blockx,blocky),dim3(16,16)>>>(D_Image , gain_r, gain_g, gain_b, orientation, width, length);
    cudaDeviceSynchronize();

    cudaMemcpy(Input,D_Image,array_size * sizeof(int),cudaMemcpyDeviceToHost);// copy data back to ram memory.

    cudaFree(D_Image);//destroy memory created in gpu.
    cudaFree(d_awbg);



    return Image;



}

PYBIND11_MODULE(awbg, m)
{
    m.def("AWBG", &AWBG, "perform Automatic White Balance Gain on the image");
}