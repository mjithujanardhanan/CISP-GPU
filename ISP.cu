/*                                                                                  ###ISP PIPELINE###

This is a cuda program with python binding which performs image signal processing pipeline operation on bayer domain images.

Operation::

    Dead Pixel Consealment ( naive algorithm. will be replaced with more optimised kernel. )
    |
    V
    Black Level Correction
    |
    V
    Lens Shading Correction ( incomplete ) 
    |
    V
    Automatic White Balance Kernels (2-pass) (gray world assumption)  :: An alternate option for user to enter the gain values are also provided (toogle the activation variable)
    |
    V
    De-Bayering Kernel (2-pass) (Hamilton adams edge directed interpolation algorithm)
    |
    V
    Color correction matrix transformation 
    |
    v
    Gamma correction (Using lookup table)
    |
    V
    processed image

*/
#include<cuda_runtime.h>
#include<stdio.h>
#include<cmath>
#include<stdlib.h>
#include<random>
#include<array>
#include <algorithm>
#include<vector>
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include<pybind11/stl.h>

#define block_dim 16                                                                                                                                                                        // Dimension of each cuda block. Each block contains 256 threads representing a pixel each
#define block_size 256  

// size of each block  
namespace py = pybind11;

__constant__ float D_MAT[9];
__constant__ int D_BLC_Offset[4];
__constant__ float D_LSC[4];

struct configuration
{
    
    int length=0;
    int width=0;
    int white_level = 65535;  //assumed 16 bit precision
    int orientation=0;

    bool DPC = true;
    int DPC_threshold=0;
    bool BLC = true;
    std::vector<int> BLC_Offset;
    bool LSC = true;
    std::vector<float> LSC_gain;
    float LSC_Max_radius=0.0f;
    bool AWB = true;
    std::vector<float> CCM_gain;
    bool CCM=true;
    bool GAMMA=true;
    float GAMMA_VALUE = 2.2f;
    std::vector<float> AWB_gain;
    bool AWB_Value_Given = false;
    bool Color_Space_Conversion = true;

};





/* this program is an execution of Defective Pixel Consealment on digital bayer domain images*/
__global__ void DP_kernel(int* Image , int* image_out, int width, int length, int threshold)                                                                                                // Image - the image on which the operation is to be performed. Image out- the output image. Threshold- the threshold for dpc correction
{

    // directional gradient calculation
    long x=blockIdx.x * blockDim.x + threadIdx.x, y=blockIdx.y * blockDim.y + threadIdx.y; // y,x are 2d image coordinates calculated from the thread id
    long idx= (y)* width  + (x);  // idx is the id of (y,x) element in the flattened image
    if(y<length && x<width) // out of bounds check
    {

        int up = (y-2)<0?y+2:y-2;               //reflective padding implementation.(important in dpc calculation)
        int down = (y+2)>=length?y-2:y+2;
        int left = (x-2)<0?x+2:x-2;
        int right=(x+2)>=width?x-2:x+2;

        int p1 = up * width + x;            //assigning ids to neighbor elements ( future updation to shared memory will remove this part)
        int p2 = up * width + left;
        int p3 = y * width + left;
        int p4 = down * width + left;
        int p5 = down * width + x;
        int p6 = down * width + right;
        int p7 = y * width + right;
        int p8 = up * width + right;


        int d1,d2,d3,d4;                    //gradient calculation
        d1 = abs(Image[p5]-Image[p1]);
        d2 = abs(Image[p6]-Image[p2]);
        d3 = abs(Image[p7]-Image[p3]);
        d4 = abs(Image[p8]-Image[p4]);

        int min=d1, neighbor_avg = (Image[p5]+Image[p1])>>1; //finding min neighbor_average. most similarity will be along the direction of least gradient.
        if(d2<min)
        {
            min=d2;
            neighbor_avg = (Image[p6]+Image[p2])>>1;
        }
        if(d3<min)
        {
            min=d3;
            neighbor_avg = (Image[p7]+Image[p3])>>1;
        }
        if(d4<min)
        {
            min=d4;
            neighbor_avg = (Image[p8]+Image[p4])>>1;
        }

        if(abs(Image[idx] -neighbor_avg) >threshold)    //DPC in effect. check against threshold to classify and replace the Dead Pixel.
        {
            image_out[idx] = neighbor_avg ;
        }

        else 
        {
            image_out[idx] = Image[idx] ;
        }

    }



}

/* this program is an execution of Black Level correction on digital bayer domain images*/
__global__ void BLC_kernel(int* Image, int width, int length)
{

    int x=blockIdx.x * blockDim.x + threadIdx.x, y=blockIdx.y * blockDim.y + threadIdx.y;
    int idx= (y)* width  + (x);
    if(y<length && x<width)
    {

       int offset[2][2]={{D_BLC_Offset[0],D_BLC_Offset[1]},{D_BLC_Offset[2],D_BLC_Offset[3]}};

       int val = Image[idx] - offset[y%2][x%2];

       Image[idx] = (val>0)?val:0;

    }



}

/* this program is an execution of Defective Pixel Consealment on digital bayer domain images*/
__global__ void LSC_kernel(int* Image , int width, int Length, float Max_radius)                                                                                                            // gain is for every color in bayer format image assed in the input configuration. 
{

    // Lens Shading Correction calculation
    long x=blockIdx.x * blockDim.x + threadIdx.x, y=blockIdx.y * blockDim.y + threadIdx.y;
    if(y<Length && x<width) //boundary check
    {
        long idx= (y)* width  + (x);
        float a[2][2]={{D_LSC[0],D_LSC[1]},{D_LSC[2],D_LSC[3]}};
        float dx = float(width)/2.0f -x; //dx distance from centre to x (here x is x)
        float dy = float(Length)/2.0f -y;//dy distance from centre to y (here y is y)
        float r = sqrtf(dx*dx + dy*dy); //radius r calculation
        
        Image[idx] = (int)(Image[idx]*( 1.0f + r*a[y%2][x%2]/ Max_radius));  /*lens shading correction modelled as a linear function (original lens shading is modelled
                                                                                as a cos^4 function which will be implemented in a future version. this is adopted only for development purpose)*/

    }



}

/* this program is an calculation of automatic white balance gain on digital bayer domain images*/
__global__ void AWBG_kernel(int* Image , unsigned long long* awbg,int orientation,int width, int length)                                                                                    // || BGGR - 0 ||  GBRG -1 || GRBG -2 || RGGB -3 ||             || awbg- || 0-> green || 1-> blue || 2-> red  ||
{
    int x=blockIdx.x * blockDim.x + threadIdx.x, y=blockIdx.y * blockDim.y + threadIdx.y;                                                                                                   // calculating the pixel coordinates y and x corresponding to the thread.
                                                                                                                                                           
    __shared__ unsigned long long Green_sum[block_size], Red_sum[block_size], Blue_sum[block_size];                                                                                      // Shared memory initialization for each channel   int x=blockIdx.x * blockDim.x + threadIdx.x, y=blockIdx.y * blockDim.y + threadIdx.y;  

    int threadid = threadIdx.y*blockDim.x + threadIdx.x;

    Green_sum[threadid]  = 0; 
    Red_sum[threadid]  = 0; 
    Blue_sum[threadid]  = 0; 

    if(y<length && x<width)
    {
        int idx= (y)* width  + (x);                                                                                                                                                     // Idx is the index of the pixel in the flattened array. 
        if(y&1 && x&1)
            {
                if(orientation ==1 || orientation ==2)
                {
                    Green_sum[threadid]  = Image[idx];
                }
                else if(orientation ==0)
                {
                    Red_sum[threadid]  =Image[idx];
                }
                else
                {
                    Blue_sum[threadid]  =Image[idx];
                }
            }
            else if(!(y&1) && !(x&1))
            {
                if(orientation ==1 || orientation ==2)
                {
                    Green_sum[threadid]  = Image[idx];
                }
                else if(orientation ==0)
                {
                    Blue_sum[threadid]  =Image[idx];
                }
                else
                {
                    Red_sum[threadid]  =Image[idx];
                }
            }
        
            else if((y&1) && !(x&1))
            {
                if(orientation ==0 || orientation ==3)
                {
                    Green_sum[threadid]  = Image[idx];
                }
                else if(orientation ==1)
                {
                    Blue_sum[threadid]  =Image[idx];
                }
                else
                {
                    Red_sum[threadid]  =Image[idx];
                }
            }
            else 
            {
                if(orientation ==0 || orientation ==3)
                {
                    Green_sum[threadid]  = Image[idx];
                }
                else if(orientation ==1)
                {
                    Red_sum[threadid]  =Image[idx];
                }
                else
                {
                    Blue_sum[threadid]  =Image[idx];
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

__global__ void AWBG_Apply_kernel(int* Image ,float gain_r,float gain_g,float gain_b, int precision, int orientation,int width, int length)                                                 // || BGGR - 0 ||  GBRG -1 || GRBG -2 || RGGB -3 ||             || awbg- || 0-> green || 1-> blue || 2-> red  ||
{
    int x=blockIdx.x * blockDim.x + threadIdx.x, y=blockIdx.y * blockDim.y + threadIdx.y;
    int idx= (y)* width  + (x);


    if(y<length && x<width)
    {
            if(y&1 && x&1)
            {
                if(orientation ==1 || orientation ==2)
                {
                    Image[idx] = max(0,min(precision, (int)roundf(Image[idx] * gain_g)));
                }
                else if(orientation ==0)
                {
                    Image[idx] = max(0,min(precision, (int)roundf(Image[idx] * gain_r)));
                }
                else
                {
                    Image[idx] = max(0,min(precision, (int)roundf(Image[idx] * gain_b)));
                }
            }
            else if(!(y&1) && !(x&1))
            {
                if(orientation ==1 || orientation ==2)
                {
                    Image[idx] = max(0,min(precision, (int)roundf(Image[idx] * gain_g)));
                }
                else if(orientation ==0)
                {
                    Image[idx] = max(0,min(precision, (int)roundf(Image[idx] * gain_b)));
                }
                else
                {
                    Image[idx] = max(0,min(precision, (int)roundf(Image[idx] * gain_r)));
                }
            }
        
            else if((y&1) && !(x&1))
            {
                if(orientation ==0 || orientation ==3)
                {
                    Image[idx] = max(0,min(precision, (int)roundf(Image[idx] * gain_g)));
                }
                else if(orientation ==1)
                {
                    Image[idx] = max(0,min(precision, (int)roundf(Image[idx] * gain_b)));
                }
                else
                {
                    Image[idx] = max(0,min(precision, (int)roundf(Image[idx] * gain_r)));
                }
            }
            else 
            {
                if(orientation ==0 || orientation ==3)
                {
                    Image[idx] = max(0,min(precision, (int)roundf(Image[idx] * gain_g)));
                }
                else if(orientation ==1)
                {
                    Image[idx] = max(0,min(precision, (int)roundf(Image[idx] * gain_r)));
                }
                else
                {
                    Image[idx] = max(0,min(precision, (int)roundf(Image[idx] * gain_b)));
                }
            }
    }

}

/* this program is an execution of edge aware interpolation of bayer domain image*/
__global__ void DEBAYER_kernel_1(int* Image , int* output, int orientation,int width, int length)                                       //green interpolation
{
    int x=blockIdx.x * block_dim + threadIdx.x, y=blockIdx.y * block_dim + threadIdx.y;                                                 //int j=blockIdx.x * block_dim + threadIdx.x, i=blockIdx.y * block_dim + threadIdx.y;
    int idx= abs(y-=2)* width  + abs(x-=2);

    __shared__ float buffer[20][21];   // in format [y][x]

    int tx=threadIdx.x, ty=threadIdx.y;  // thread x and thread y
    buffer[ty][tx] =0;
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

/* this program is for applying transform matrix to the color image*/
__global__ void Transform_Kernel(int* channel_1, int* channel_2, int* channel_3, int precision,  int width, int length)
{
    int x=blockIdx.x * blockDim.x + threadIdx.x, y=blockIdx.y * blockDim.y + threadIdx.y;
    
    
    if(y<length && x<width)
    {
        int idx= (y)* width  + (x); 

        int Temp_c1 =    channel_1[idx];
        int Temp_c2 =    channel_2[idx];
        int Temp_c3 =    channel_3[idx];
        int C1,C2,C3;

        C1 =      roundf(D_MAT[0] * Temp_c1 + D_MAT[1] * Temp_c2 + D_MAT[2] * Temp_c3);
        C2 =      roundf(D_MAT[3] * Temp_c1 + D_MAT[4] * Temp_c2 + D_MAT[5] * Temp_c3);
        C3 =      roundf(D_MAT[6] * Temp_c1 + D_MAT[7] * Temp_c2 + D_MAT[8] * Temp_c3);

        channel_1[idx] =          max(0, min(precision, C1));
        channel_2[idx] =        max(0, min(precision, C2));
        channel_3[idx] =         max(0, min(precision, C3));
    }
}

// matrix multiplication function to multiply continuous linear transformations together
void MATRIX_MULTIPLICATION(float* matrix_1, const float* matrix_2, int dim)
{
    std::vector<float> output(dim * dim, 0.0f);
    for(int i=0;i<dim;i++)
    {
        for(int j=0;j<dim;j++)
        {
            for(int k=0;k<dim;k++)
            {
                output[i * dim + j] += matrix_1[i*dim + k] * matrix_2[k*dim + j] ;
            }
               
        }
        
    }

    for(int i = 0; i<dim * dim ; i++)
        matrix_1[i] = output[i];

}

/* This program is for applying gamma correction to the image*/
__global__ void GAMMA_kernel( int* red, int* green, int* blue, unsigned char* LUT, int width, int length)
{
    int x=blockIdx.x * block_dim + threadIdx.x, y=blockIdx.y * block_dim + threadIdx.y;
    int idx= (y)* width  + (x);  
    if(y<length && x<width)
    {
        green[idx] = LUT[green[idx]];
        red[idx] = LUT[red[idx]];
        blue[idx] = LUT[blue[idx]];
    }
}

// Host code for Image Signal Processing Pipeline
py::tuple ISP(py::array_t<int> Image, const configuration& cfg )    
{
    ////////////////////////////////////////////////////
    //
    //
    // ### loading input image ###
    //
    //
    ////////////////////////////////////////////////////
    auto buffer = Image.request();
    if(buffer.ndim != 1)            //error check to see if flattened image is passed.
        throw std::runtime_error("image must be Flattened :: ISP module");
    if(buffer.size < (cfg.width * cfg.length))
        throw std::runtime_error("Wrong image size :: ISP module");
    int *Input = static_cast<int*>(buffer.ptr);
    long array_size = static_cast<int>(buffer.size);

    ////////////////////////////////////////////////////
    //
    //
    // ### creating Device variables for loading data to GPU. D_Image_1 and D_Image_2 are image input output pairs which change job after each kernel. ###
    //
    //
    ////////////////////////////////////////////////////
    int *D_Image_1, *D_image_2; 
    cudaMalloc( &D_Image_1, array_size * sizeof(int)); // creating memory pointers on gpu memory for image.
    cudaMalloc( &D_image_2, array_size * sizeof(int));

    ////////////////////////////////////////////////////
    //
    //
    // ### defining no of blocks for the GPU kernel. each block will have 256 threads, ie. 16 by 16 block. this ensures maximum occupency of the multiprocessor. ###
    //
    //
    ////////////////////////////////////////////////////
    const int blockx= (cfg.width%16 == 0)?(cfg.width/16):(cfg.width/16 +1),blocky= (cfg.length%16 == 0)?(cfg.length/16):(cfg.length/16 +1);

    ////////////////////////////////////////////////////
    //
    //
    // ### loading input image into gpu to begin the kernel. ###
    //
    //
    ////////////////////////////////////////////////////
    cudaMemcpy(D_Image_1 , Input, array_size * sizeof(int), cudaMemcpyHostToDevice);

    ////////////////////////////////////////////////////
    //
    //
    // ### pipeline starts. ###
    // Executing Dead Pixel Correction Kernel
    //
    ////////////////////////////////////////////////////
    if(cfg.DPC)
    {
        DP_kernel<<<dim3(blockx,blocky),dim3(block_dim,block_dim)>>>(D_Image_1,D_image_2, cfg.width, cfg.length, cfg.DPC_threshold);          //calling __global__ function (CUDA kernel)
        cudaDeviceSynchronize(); //wait until all kernels stop executing.
    }
    else
    {
        cudaMemcpy(D_image_2,D_Image_1, array_size * sizeof(int), cudaMemcpyDeviceToDevice);
    }

    ////////////////////////////////////////////////////
    //
    //
    // D_Image_1 is not used after this point therefore the memory is released.
    //
    //
    ////////////////////////////////////////////////////
    cudaFree(D_Image_1);

    ////////////////////////////////////////////////////
    //
    //
    // Executing Black Level Correction Kernel
    //
    //
    ////////////////////////////////////////////////////
    if(cfg.BLC)
    {

        if(cfg.BLC_Offset.size() != 4)
        {
            throw std::runtime_error("BLC Offset must contain exactly 4 positive integer values");
        }
        cudaMemcpyToSymbol( D_BLC_Offset, cfg.BLC_Offset.data(), 4 * sizeof(int));

        BLC_kernel<<<dim3(blockx,blocky),dim3(block_dim,block_dim)>>>(D_image_2,cfg.width,cfg.length);
        cudaDeviceSynchronize();
    }

    ////////////////////////////////////////////////////
    //
    //
    // Executing Lens Shading Correction
    //
    //
    ////////////////////////////////////////////////////
    if(cfg.LSC)
    {
        if(cfg.LSC_gain.size() != 4)
        {
            throw std::runtime_error("LSC Gain must contain exactly 4 float values");
        }
        cudaMemcpyToSymbol( D_LSC, cfg.LSC_gain.data(), 4 * sizeof(float));
        LSC_kernel<<<dim3(blockx,blocky),dim3(block_dim,block_dim)>>>(D_image_2,cfg.width,cfg.length,cfg.LSC_Max_radius);
        cudaDeviceSynchronize();

    }

    ////////////////////////////////////////////////////
    //
    //
    // Executing Automatic white balance gain
    //
    //
    ////////////////////////////////////////////////////
    if(cfg.AWB)
    {
        float GAIN_RED;
        float GAIN_GREEN;
        float GAIN_BLUE;
        if(!cfg.AWB_Value_Given)
        {
            unsigned long long* D_AWBG;
            unsigned long long H_AWBG[3];

            cudaMalloc( &D_AWBG, 3 * sizeof(unsigned long long));
            cudaMemset( D_AWBG, 0, 3 * sizeof(unsigned long long));

            AWBG_kernel<<<dim3(blockx,blocky),dim3(block_dim,block_dim)>>>( D_image_2, D_AWBG, cfg.orientation, cfg.width, cfg.length);
            cudaDeviceSynchronize();

            cudaMemcpy(H_AWBG, D_AWBG, 3 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

            GAIN_RED = (float)(double(H_AWBG[0])/double(2*H_AWBG[2]));
            GAIN_GREEN = 1.0f;
            GAIN_BLUE = (float)(double(H_AWBG[0])/double(2*H_AWBG[1]));
            cudaFree(D_AWBG);

        }
        else
        {
            GAIN_RED = cfg.AWB_gain[0];
            GAIN_GREEN = cfg.AWB_gain[1];
            GAIN_BLUE = cfg.AWB_gain[2];
        }
        AWBG_Apply_kernel<<<dim3(blockx,blocky),dim3(block_dim,block_dim)>>>( D_image_2, GAIN_RED, GAIN_GREEN, GAIN_BLUE, cfg.white_level, cfg.orientation, cfg.width, cfg.length);
        cudaDeviceSynchronize();
        
    }

    ////////////////////////////////////////////////////
    //
    //
    //Executing De-Bayer kernels
    //This is not an optional kernel as debayering produces 3 color channel output. This stage is required for the following stages.
    //                      Channel_0 :: Red   Channel_1 :: Green    Channel_2 :: Red
    //
    ////////////////////////////////////////////////////
    int* CHANNEL_0;
    int* CHANNEL_1;
    int* CHANNEL_2;

    cudaMalloc(&CHANNEL_0, array_size * sizeof(int));
    cudaMalloc(&CHANNEL_1, array_size * sizeof(int));
    cudaMalloc(&CHANNEL_2, array_size * sizeof(int));

    DEBAYER_kernel_1<<<dim3(blockx,blocky),dim3(block_dim+4 ,block_dim+4)>>>(D_image_2, CHANNEL_1, cfg.orientation, cfg.width, cfg.length); // the no of threads are increased to 400 per block for acting as halo and padding for the block level operations
    cudaDeviceSynchronize();
    DEBAYER_kernel_2<<<dim3(blockx,blocky),dim3(block_dim+4 ,block_dim+4)>>>(D_image_2, CHANNEL_1, CHANNEL_0, CHANNEL_2, cfg.orientation, cfg.width, cfg.length);
    cudaDeviceSynchronize();

    ////////////////////////////////////////////////////
    //
    //
    //freeing D_image_2 as the source image is no longer required.
    //
    //
    ////////////////////////////////////////////////////
    cudaFree(D_image_2);

    ////////////////////////////////////////////////////
    //
    //
    //Executing Color Correction calculations using CCM matrix.
    //
    //
    ////////////////////////////////////////////////////
    if(cfg.CCM)
    {
        if(cfg.CCM_gain.size() != 9)
        {
            throw std::runtime_error("CCM must contain exactly 9 floats");
        }

        cudaMemcpyToSymbol( D_MAT, cfg.CCM_gain.data(), 9 * sizeof(float));
        Transform_Kernel<<<dim3(blockx,blocky),dim3(block_dim,block_dim)>>>(CHANNEL_0, CHANNEL_1, CHANNEL_2, cfg.white_level, cfg.width, cfg.length);
        cudaDeviceSynchronize();
    }

    ////////////////////////////////////////////////////
    //
    //
    // Executing Gamma correction using Lookup Table
    //
    //
    ////////////////////////////////////////////////////
    if(cfg.GAMMA)
    {
        unsigned char *D_LUT;
        float x;

        cudaMalloc(&D_LUT, (cfg.white_level+1) * sizeof(unsigned char));

        std::vector<unsigned char> LUT(cfg.white_level+1);

        for(int i=0;i<(cfg.white_level+1);i++)
        {
            x = float(i)/float(cfg.white_level);
            x = powf(x, (1.0f/cfg.GAMMA_VALUE));

            LUT[i] = (int)roundf(x*255);
        }

        cudaMemcpy(D_LUT, LUT.data() , (cfg.white_level+1) *  sizeof(unsigned char), cudaMemcpyHostToDevice);
        GAMMA_kernel<<<dim3(blockx,blocky),dim3(block_dim,block_dim)>>>( CHANNEL_0, CHANNEL_1, CHANNEL_2, D_LUT, cfg.width, cfg.length);
        cudaDeviceSynchronize();
        cudaFree(D_LUT);
    }

    ////////////////////////////////////////////////////
    //
    //
    // Color Space Conversion to Yuv
    //  Channel_0 :: Y   Channel_1 :: U    Channel_2 :: V
    //
    ////////////////////////////////////////////////////
    if(cfg.Color_Space_Conversion)
    {
        float CSC[9] = {0.2988, 0.5869, 0.1143, -0.1689, -0.3311, 0.5000, 0.5000, -0.4189, -0.0811};
        cudaMemcpyToSymbol( D_MAT, CSC, 9 * sizeof(float));
        Transform_Kernel<<<dim3(blockx,blocky),dim3(block_dim,block_dim)>>>(CHANNEL_0, CHANNEL_1, CHANNEL_2, cfg.white_level, cfg.width, cfg.length);
        cudaDeviceSynchronize();
    }



    ////////////////////////////////////////////////////
    //
    //
    // Color Space Conversion to RGB
    //  Channel_0 :: R   Channel_1 :: G    Channel_2 :: B
    //
    ////////////////////////////////////////////////////
    if(cfg.Color_Space_Conversion)
    {
        float CSC[9] = {1, -0.0012, 1.402, 1, -0.3444, -0.7141, 1, 1.772, 0.0008241};
        cudaMemcpyToSymbol( D_MAT, CSC, 9 * sizeof(float));
        Transform_Kernel<<<dim3(blockx,blocky),dim3(block_dim,block_dim)>>>(CHANNEL_0, CHANNEL_1, CHANNEL_2, cfg.white_level, cfg.width, cfg.length);
        cudaDeviceSynchronize();
    }


    py::array_t<int> Red({cfg.length, cfg.width});
    py::array_t<int> Green({cfg.length, cfg.width});
    py::array_t<int> Blue({cfg.length, cfg.width});

    auto r_buf = Red.request();
    auto g_buf = Green.request();
    auto b_buf = Blue.request();

    int* R = static_cast<int*>(r_buf.ptr);
    int* G = static_cast<int*>(g_buf.ptr);
    int* B = static_cast<int*>(b_buf.ptr);

    cudaMemcpy(R, CHANNEL_0, array_size*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(G, CHANNEL_1, array_size*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(B, CHANNEL_2, array_size*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(CHANNEL_0);
    cudaFree(CHANNEL_1);
    cudaFree(CHANNEL_2);

    return py::make_tuple(Red, Green, Blue);
    
}

PYBIND11_MODULE(ISP, m) {
    // 1. Bind the configuration struct
    py::class_<configuration>(m, "Configuration",
        R"pbdoc(
    ISP Configuration Structure

    Attributes
    ----------
    width : int
        Image width.

    length : int
        Image height.

    white_level : int
        Maximum sensor value.

    orientation : int
        Bayer pattern:
        0 = BGGR
        1 = GBRG
        2 = GRBG
        3 = RGGB

    DPC : bool
        Enable dead pixel correction.

    DPC_threshold : int
        Threshold for dead pixel correction

    BLC : bool
        Enable black level correction 

    BLC_Offset : int vector | 4 values
        offset for black level correction in order (00 , 01, 10, 11)

    LSC : bool
        Enable lens shading correction.

    LSC_gain : float vector | 4 values
        gain values for leans shading correction in order (00 , 01, 10, 11)
    
    LSC_Max_radius : float
        max value for lens shading correction radius

    AWB : bool
        Enable auto white balance.

    CCM : bool
        Enable color correction matrix.
    
    CCM_gain : float vector : 9 values
        color correction matrix:: as flattened array in order [00 01 02 10 11 12 20 21 22]

    GAMMA : bool
        Enable gamma correction.
    
    GAMMA_VALUE : float
        value for gamma correction.
    
    AWB_gain : float vector
        value for AWG gain.
    
    AWB_value : bool
        if automatic correction or not

    )pbdoc")
        .def(py::init<>())
        .def_readwrite("length", &configuration::length)
        .def_readwrite("width", &configuration::width)
        .def_readwrite("white_level", &configuration::white_level)
        .def_readwrite("DPC", &configuration::DPC)
        .def_readwrite("DPC_threshold", &configuration::DPC_threshold)
        .def_readwrite("BLC", &configuration::BLC)
        .def_readwrite("BLC_Offset", &configuration::BLC_Offset)
        .def_readwrite("LSC", &configuration::LSC)
        .def_readwrite("LSC_gain", &configuration::LSC_gain)
        .def_readwrite("LSC_Max_radius", &configuration::LSC_Max_radius)
        .def_readwrite("orientation", &configuration::orientation)
        .def_readwrite("AWB", &configuration::AWB)
        .def_readwrite("CCM_gain", &configuration::CCM_gain)
        .def_readwrite("CCM", &configuration::CCM)
        .def_readwrite("GAMMA", &configuration::GAMMA)
        .def_readwrite("GAMMA_VALUE", &configuration::GAMMA_VALUE)
        .def_readwrite("AWB_gain", &configuration::AWB_gain)
        .def_readwrite("AWB_Value_Given", &configuration::AWB_Value_Given)
        .def_readwrite("Color_Space_Conversion", &configuration::Color_Space_Conversion);
        
        Color_Space_Conversion

    // 2. Bind your functions
    // Note: If LSC takes the struct as an argument, define it like this:
    m.def("ISP", &ISP, "Image Signal Processing Pipeline");
}