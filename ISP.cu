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
#define working_precision 65535

// size of each block  
namespace py = pybind11;

__constant__ float D_MAT[9];
__constant__ float D_BLC_Offset[4];
__constant__ float D_LSC[4];

struct configuration
{
    
    int     length=0;
    int     width=0;
    float   white_level = 65535.0f;  //assumed 16 bit precision
    int     orientation=0;

    bool    DPC = true;
    float   DPC_threshold=0;

    bool    BLC = true;
    std::vector<float> BLC_Offset;

    bool    LSC = true;
    std::vector<float> LSC_gain;
    float   LSC_Max_radius=0.0f;

    bool    AWB = true;
    std::vector<float> AWB_gain;
    bool    AWB_Value_Given = false;

    std::vector<float> CCM_gain;
    bool    CCM=true;

    bool    Color_Space_Conversion = true;
    bool    Brightness = false;
    float   Brightness_value = 1.0f;
    bool    Saturation = false;
    float   Saturation_value = 1.0f;

    bool    GAMMA=true;
    float   GAMMA_VALUE = 2.2f;

    bool    Bilateral_Filter = false;
    int     Bilateral_kernel_size = 3;
    float   Bilateral_Domain_STD = 10.0f;
    float   Bilateral_Range_STD = 10.0f;

    
    
};


/* this program is an execution of Defective Pixel Consealment on digital bayer domain images*/
__global__ void DP_kernel(float* Image , float* image_out, int width, int length, float threshold)                                                                                                // Image - the image on which the operation is to be performed. Image out- the output image. Threshold- the threshold for dpc correction
{

    // directional gradient calculation
    int x=blockIdx.x * blockDim.x + threadIdx.x, y=blockIdx.y * blockDim.y + threadIdx.y; // y,x are 2d image coordinates calculated from the thread id
    int idx= (y)* width  + (x);  // idx is the id of (y,x) element in the flattened image
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


        float d1,d2,d3,d4;                    //gradient calculation
        d1 = fabsf(Image[p5]-Image[p1]);
        d2 = fabsf(Image[p6]-Image[p2]);
        d3 = fabsf(Image[p7]-Image[p3]);
        d4 = fabsf(Image[p8]-Image[p4]);

        float min=d1, neighbor_avg = (Image[p5]+Image[p1])* 0.5f; //finding min neighbor_average. most similarity will be along the direction of least gradient.
        if(d2<min)
        {
            min=d2;
            neighbor_avg = (Image[p6]+Image[p2])* 0.5f;
        }
        if(d3<min)
        {
            min=d3;
            neighbor_avg = (Image[p7]+Image[p3])* 0.5f;
        }
        if(d4<min)
        {
            min=d4;
            neighbor_avg = (Image[p8]+Image[p4])* 0.5f;
        }

        if(fabsf(Image[idx] -neighbor_avg) >threshold)    //DPC in effect. check against threshold to classify and replace the Dead Pixel.
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
__global__ void BLC_kernel(float* Image, int width, int length)
{

    int x=blockIdx.x * blockDim.x + threadIdx.x, y=blockIdx.y * blockDim.y + threadIdx.y;
    int idx= (y)* width  + (x);
    if(y<length && x<width)
    {

       float offset[2][2]={{D_BLC_Offset[0],D_BLC_Offset[1]},{D_BLC_Offset[2],D_BLC_Offset[3]}};

       Image[idx] = (Image[idx] - offset[y%2][x%2]);

    }



}

/* this program is an execution of Defective Pixel Consealment on digital bayer domain images*/
__global__ void LSC_kernel(float* Image , int width, int Length, float Max_radius)                                                                                                            // gain is for every color in bayer format image assed in the input configuration. 
{

    // Lens Shading Correction calculation
    int x=blockIdx.x * blockDim.x + threadIdx.x, y=blockIdx.y * blockDim.y + threadIdx.y;
    if(y<Length && x<width) //boundary check
    {
        int idx= (y)* width  + (x);
        float a[2][2]={{D_LSC[0],D_LSC[1]},{D_LSC[2],D_LSC[3]}};
        float dx = float(width)/2.0f  -(float)x; //dx distance from centre to x (here x is x)
        float dy = float(Length)/2.0f -(float)y;//dy distance from centre to y (here y is y)
        float r = sqrtf(dx*dx + dy*dy); //radius r calculation
        
        Image[idx] = Image[idx]*( 1.0f + r*a[y%2][x%2]/ Max_radius);  /*lens shading correction modelled as a linear function (original lens shading is modelled
                                                                                as a cos^4 function which will be implemented in a future version. this is adopted only for development purpose)*/

    }



}

/* this program is an calculation of automatic white balance gain on digital bayer domain images*/
__global__ void AWBG_kernel(float* Image , double* awbg,int orientation,int width, int length)                                                                                    // || BGGR - 0 ||  GBRG -1 || GRBG -2 || RGGB -3 ||             || awbg- || 0-> green || 1-> blue || 2-> red  ||
{
    int x=blockIdx.x * blockDim.x + threadIdx.x, y=blockIdx.y * blockDim.y + threadIdx.y;                                                                                                   // calculating the pixel coordinates y and x corresponding to the thread.
                                                                                                                                                           
    __shared__ double Green_sum[block_size], Red_sum[block_size], Blue_sum[block_size];                                                                                      // Shared memory initialization for each channel   int x=blockIdx.x * blockDim.x + threadIdx.x, y=blockIdx.y * blockDim.y + threadIdx.y;  

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

__global__ void AWBG_Apply_kernel(float* Image ,float gain_r,float gain_g,float gain_b, int orientation,int width, int length)                                                 // || BGGR - 0 ||  GBRG -1 || GRBG -2 || RGGB -3 ||             || awbg- || 0-> green || 1-> blue || 2-> red  ||
{
    int x=blockIdx.x * blockDim.x + threadIdx.x, y=blockIdx.y * blockDim.y + threadIdx.y;
    int idx= (y)* width  + (x);


    if(y<length && x<width)
    {
            if(y&1 && x&1)
            {
                if(orientation ==1 || orientation ==2)
                {
                    Image[idx] = Image[idx] * gain_g;
                }
                else if(orientation ==0)
                {
                    Image[idx] = Image[idx] * gain_r;
                }
                else
                {
                    Image[idx] = Image[idx] * gain_b;
                }
            }
            else if(!(y&1) && !(x&1))
            {
                if(orientation ==1 || orientation ==2)
                {
                    Image[idx] = Image[idx] * gain_g;
                }
                else if(orientation ==0)
                {
                    Image[idx] = Image[idx] * gain_b;
                }
                else
                {
                    Image[idx] = Image[idx] * gain_r;
                }
            }
        
            else if((y&1) && !(x&1))
            {
                if(orientation ==0 || orientation ==3)
                {
                    Image[idx] = Image[idx] * gain_g;
                }
                else if(orientation ==1)
                {
                    Image[idx] = Image[idx] * gain_b;
                }
                else
                {
                    Image[idx] = Image[idx] * gain_r;
                }
            }
            else 
            {
                if(orientation ==0 || orientation ==3)
                {
                    Image[idx] = Image[idx] * gain_g;
                }
                else if(orientation ==1)
                {
                    Image[idx] = Image[idx] * gain_r;
                }
                else
                {
                    Image[idx] = Image[idx] * gain_b;
                }
            }
    }

}

/* this program is an execution of edge aware interpolation of bayer domain image*/
__global__ void DEBAYER_kernel_1(float* Image , float* output, int orientation,int width, int length)                                       //green interpolation
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
                        output[idx] = ((buffer[ty-1][tx] + buffer[ty+1][tx])*0.5 + (2* buffer[ty][tx] -(buffer[ty-2][tx] + buffer[ty+2][tx]))*0.25f);
                    }
                    
                    else if (dh<dv)
                    {
                        output[idx] = ((buffer[ty][tx-1] + buffer[ty][tx+1])*0.5 + (2* buffer[ty][tx] -(buffer[ty][tx-2] + buffer[ty][tx+2]))*0.25f);
                    }
                    else
                    {
                        output[idx] = ((buffer[ty-1][tx] + buffer[ty+1][tx] + buffer[ty][tx-1] + buffer[ty][tx+1])*0.25 + (2* buffer[ty][tx] -(buffer[ty-2][tx] + buffer[ty+2][tx]) +2* buffer[ty][tx] -(buffer[ty][tx-2] + buffer[ty][tx+2]))*0.125f);
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
                        output[idx] = ((buffer[ty-1][tx] + buffer[ty+1][tx])*0.5 + (2* buffer[ty][tx] -(buffer[ty-2][tx] + buffer[ty+2][tx]))*0.25f);
                    }
                    
                    else if (dh<dv)
                    {
                        output[idx] = ((buffer[ty][tx-1] + buffer[ty][tx+1])*0.5 + (2* buffer[ty][tx] -(buffer[ty][tx-2] + buffer[ty][tx+2]))*0.25f);
                    }
                    else
                    {
                        output[idx] = ((buffer[ty-1][tx] + buffer[ty+1][tx] + buffer[ty][tx-1] + buffer[ty][tx+1])*0.25 + (2* buffer[ty][tx] -(buffer[ty-2][tx] + buffer[ty+2][tx]) +2* buffer[ty][tx] -(buffer[ty][tx-2] + buffer[ty][tx+2]))*0.125f);
                    }
                    
                }
            }
        }
    }
}

__global__ void DEBAYER_kernel_2(float* Image , float* green, float* red, float* blue, int orientation,int width, int length)                   //color interpolation
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
                            red[idx] = (bufferg[ty][tx]+0.5f *(buffer1[ty-1][tx]-bufferg[ty-1][tx] + (buffer1[ty+1][tx]-bufferg[ty+1][tx])));                     //vertical interpolation
                            blue[idx]= (bufferg[ty][tx]+0.5f *(buffer1[ty][tx-1]-bufferg[ty][tx-1] + (buffer1[ty][tx+1]-bufferg[ty][tx+1])));                     //horizontal interpolation
                        }
                        else
                        {
                            blue[idx] = (bufferg[ty][tx]+0.5f *(buffer1[ty-1][tx]-bufferg[ty-1][tx] + (buffer1[ty+1][tx]-bufferg[ty+1][tx])));                    //vertical interpolation
                            red[idx]  = (bufferg[ty][tx]+0.5f *(buffer1[ty][tx-1]-bufferg[ty][tx-1] + (buffer1[ty][tx+1]-bufferg[ty][tx+1])));                    //horizontal interpolation
                        }
                    }
                    else
                    {
                        if(x&1 && !(y&1))
                        {
                            blue[idx] = (bufferg[ty][tx]+0.5f *(buffer1[ty-1][tx]-bufferg[ty-1][tx] + (buffer1[ty+1][tx]-bufferg[ty+1][tx])));                    //vertical interpolation
                            red[idx]  = (bufferg[ty][tx]+0.5f *(buffer1[ty][tx-1]-bufferg[ty][tx-1] + (buffer1[ty][tx+1]-bufferg[ty][tx+1])));                    //horizontal interpolation
                            
                        }
                        else
                        {
                            red[idx] = (bufferg[ty][tx]+0.5f *(buffer1[ty-1][tx]-bufferg[ty-1][tx] + (buffer1[ty+1][tx]-bufferg[ty+1][tx])));                     //vertical interpolation
                            blue[idx]= (bufferg[ty][tx]+0.5f *(buffer1[ty][tx-1]-bufferg[ty][tx-1] + (buffer1[ty][tx+1]-bufferg[ty][tx+1])));                     //horizontal interpolation
                        }
                    }
                }
                else
                {
                    if(orientation ==0)
                    {
                        if(y&1 && x&1)
                        {
                            red[idx] = (buffer1[ty][tx]);
                            blue[idx] =(bufferg[ty][tx]+0.25f*(buffer1[ty-1][tx-1]-bufferg[ty-1][tx-1] + buffer1[ty+1][tx+1]-bufferg[ty+1][tx+1] +buffer1[ty-1][tx+1]-bufferg[ty-1][tx+1] + buffer1[ty+1][tx-1]-bufferg[ty+1][tx-1]));
                        }
                        else 
                        {
                            blue[idx] =(buffer1[ty][tx]);
                            red[idx]  =(bufferg[ty][tx]+0.25f*(buffer1[ty-1][tx-1]-bufferg[ty-1][tx-1] + buffer1[ty+1][tx+1]-bufferg[ty+1][tx+1] +buffer1[ty-1][tx+1]-bufferg[ty-1][tx+1] + buffer1[ty+1][tx-1]-bufferg[ty+1][tx-1]));
                        }
            
                    }
                    else
                    {
                        if(y&1 && x&1)
                        {
                            blue[idx] =(buffer1[ty][tx]);
                            red[idx]  =(bufferg[ty][tx]+0.25f*(buffer1[ty-1][tx-1]-bufferg[ty-1][tx-1] + buffer1[ty+1][tx+1]-bufferg[ty+1][tx+1] +buffer1[ty-1][tx+1]-bufferg[ty-1][tx+1] + buffer1[ty+1][tx-1]-bufferg[ty+1][tx-1]));
                        }
                        else 
                        {
                            red[idx]  =(buffer1[ty][tx]);
                            blue[idx] =(bufferg[ty][tx]+0.25f*(buffer1[ty-1][tx-1]-bufferg[ty-1][tx-1] + buffer1[ty+1][tx+1]-bufferg[ty+1][tx+1] +buffer1[ty-1][tx+1]-bufferg[ty-1][tx+1] + buffer1[ty+1][tx-1]-bufferg[ty+1][tx-1]));
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
                            blue[idx] = (bufferg[ty][tx]+0.5f *(buffer1[ty-1][tx]-bufferg[ty-1][tx] + (buffer1[ty+1][tx]-bufferg[ty+1][tx])));                    //vertical interpolation
                            red[idx]  = (bufferg[ty][tx]+0.5f *(buffer1[ty][tx-1]-bufferg[ty][tx-1] + (buffer1[ty][tx+1]-bufferg[ty][tx+1])));                    //horizontal interpolation
                            
                        }
                        else
                        {
                            red[idx] = (bufferg[ty][tx]+0.5f *(buffer1[ty-1][tx]-bufferg[ty-1][tx] + (buffer1[ty+1][tx]-bufferg[ty+1][tx])));                     //vertical interpolation
                            blue[idx]= (bufferg[ty][tx]+0.5f *(buffer1[ty][tx-1]-bufferg[ty][tx-1] + (buffer1[ty][tx+1]-bufferg[ty][tx+1])));                     //horizontal interpolation
                        }
                    }
                    else
                    {
                        if(x&1 && y&1)
                        {
                            red[idx] = (bufferg[ty][tx]+0.5f *(buffer1[ty-1][tx]-bufferg[ty-1][tx] + (buffer1[ty+1][tx]-bufferg[ty+1][tx])));                     //vertical interpolation
                            blue[idx]= (bufferg[ty][tx]+0.5f *(buffer1[ty][tx-1]-bufferg[ty][tx-1] + (buffer1[ty][tx+1]-bufferg[ty][tx+1])));                     //horizontal interpolation
                        }
                        else
                        {
                            blue[idx] = (bufferg[ty][tx]+0.5f *(buffer1[ty-1][tx]-bufferg[ty-1][tx] + (buffer1[ty+1][tx]-bufferg[ty+1][tx])));                    //vertical interpolation
                            red[idx]  = (bufferg[ty][tx]+0.5f *(buffer1[ty][tx-1]-bufferg[ty][tx-1] + (buffer1[ty][tx+1]-bufferg[ty][tx+1])));                    //horizontal interpolation
                        }
                    }
                }
                else
                {
                    if(orientation ==1)
                    {
                        if(!(y&1) && x&1)
                        {
                            blue[idx] = (buffer1[ty][tx]);
                            red[idx]  = (bufferg[ty][tx]+0.25f*(buffer1[ty-1][tx-1]-bufferg[ty-1][tx-1] + buffer1[ty+1][tx+1]-bufferg[ty+1][tx+1] +buffer1[ty-1][tx+1]-bufferg[ty-1][tx+1] + buffer1[ty+1][tx-1]-bufferg[ty+1][tx-1]));
                        }
                        else 
                        {
                            red[idx]  = (buffer1[ty][tx]);
                            blue[idx] = (bufferg[ty][tx]+0.25f*(buffer1[ty-1][tx-1]-bufferg[ty-1][tx-1] + buffer1[ty+1][tx+1]-bufferg[ty+1][tx+1] +buffer1[ty-1][tx+1]-bufferg[ty-1][tx+1] + buffer1[ty+1][tx-1]-bufferg[ty+1][tx-1]));
                        }
            
                    }
                    else
                    {
                        if(!(y&1) && x&1)
                        {
                            red[idx]  = (buffer1[ty][tx]);
                            blue[idx] = (bufferg[ty][tx]+0.25f*(buffer1[ty-1][tx-1]-bufferg[ty-1][tx-1] + buffer1[ty+1][tx+1]-bufferg[ty+1][tx+1] +buffer1[ty-1][tx+1]-bufferg[ty-1][tx+1] + buffer1[ty+1][tx-1]-bufferg[ty+1][tx-1]));
                        } 
                        else 
                        {
                            blue[idx] = (buffer1[ty][tx]);
                            red[idx]  = (bufferg[ty][tx]+0.25f*(buffer1[ty-1][tx-1]-bufferg[ty-1][tx-1] + buffer1[ty+1][tx+1]-bufferg[ty+1][tx+1] +buffer1[ty-1][tx+1]-bufferg[ty-1][tx+1] + buffer1[ty+1][tx-1]-bufferg[ty+1][tx-1]));
                        }


                    }
                    
                }
            }
        }
    }
}

/* this program is for applying transform matrix to the color image*/
__global__ void Transform_Kernel(float* channel_1, float* channel_2, float* channel_3,  int width, int length)
{
    int x=blockIdx.x * blockDim.x + threadIdx.x, y=blockIdx.y * blockDim.y + threadIdx.y;
    
    
    if(y<length && x<width)
    {
        int idx= (y)* width  + (x); 

        float Temp_c1 =    channel_1[idx];
        float Temp_c2 =    channel_2[idx];
        float Temp_c3 =    channel_3[idx];

        channel_1[idx] =     (D_MAT[0] * Temp_c1 + D_MAT[1] * Temp_c2 + D_MAT[2] * Temp_c3);
        channel_2[idx]=      (D_MAT[3] * Temp_c1 + D_MAT[4] * Temp_c2 + D_MAT[5] * Temp_c3);
        channel_3[idx]=      (D_MAT[6] * Temp_c1 + D_MAT[7] * Temp_c2 + D_MAT[8] * Temp_c3);


    }
}

/* This program is for applying Mapping using a Look Up Table(LUT) to the image on all channels*/
__global__ void LUT_kernel( float* red, float* green, float* blue, float* LUT, float white_value, int width, int length)
{
    int x=blockIdx.x * block_dim + threadIdx.x, y=blockIdx.y * block_dim + threadIdx.y;
    int idx= (y)* width  + (x);  
    if(y<length && x<width)
    {
        green[idx] = LUT[(int)roundf(fmaxf(0.0f,fminf(white_value, green[idx])))];
        red[idx] = LUT[(int)roundf(fmaxf(0.0f,fminf(white_value, red[idx])))];
        blue[idx] = LUT[(int)roundf(fmaxf(0.0f,fminf(white_value, blue[idx])))];
    }
}

/* This program is for applying Mapping using a Look Up Table(LUT) to the image on one channel*/
__global__ void LUT_kernel( float* Channel, float* LUT, float white_value, int width, int length)
{
    int x=blockIdx.x * block_dim + threadIdx.x, y=blockIdx.y * block_dim + threadIdx.y;
    int idx= (y)* width  + (x);  
    if(y<length && x<width)
    {
        Channel[idx] = LUT[(int)roundf(fmaxf(0.0f,fminf(white_value,(Channel[idx]))))];
    }
}

/* This program is for scaling to the image on one channel*/
__global__ void Scaling_kernel( float* Channel,  float Scaling_factor, int width, int length)
{
    int x=blockIdx.x * block_dim + threadIdx.x, y=blockIdx.y * block_dim + threadIdx.y;
    int idx= (y)* width  + (x);  
    if(y<length && x<width)
    {
        Channel[idx] = (Channel[idx] * Scaling_factor);
    }
}

__global__ void Scaling_kernel_2Channel( float* channel_1, float* channel_2, float Scaling_factor, int width, int length)
{
    int x=blockIdx.x * block_dim + threadIdx.x, y=blockIdx.y * block_dim + threadIdx.y;
    int idx= (y)* width  + (x);  
    if(y<length && x<width)
    {
        channel_1[idx] = (channel_1[idx] * Scaling_factor);
        channel_2[idx] = (channel_2[idx] * Scaling_factor);
    }
}

__global__ void Scaling_kernel_3Channel( float* channel_1, float* channel_2, float* channel_3, float Scaling_factor_1, float Scaling_factor_2, int width, int length)
{
    int x=blockIdx.x * block_dim + threadIdx.x, y=blockIdx.y * block_dim + threadIdx.y;
    int idx= (y)* width  + (x);  
    if(y<length && x<width)
    {
        channel_1[idx] = (channel_1[idx] * Scaling_factor_1);
        channel_2[idx] = (channel_2[idx] * Scaling_factor_2);
        channel_3[idx] = (channel_3[idx] * Scaling_factor_2);
    }
}

/* This program is for converting datatype of image to 8-bit integer*/
__global__ void Norm_kernel( float* red, float* green, float* blue,int* rint, int* gint, int*bint, float white_value, int width, int length)
{
    int x=blockIdx.x * block_dim + threadIdx.x, y=blockIdx.y * block_dim + threadIdx.y;
    int idx= (y)* width  + (x);  
    if(y<length && x<width)
    {
        gint[idx] =    (int)roundf(fmaxf(0.0f,fminf(255.0,green[idx]*255.0f/white_value)));
        rint[idx] =    (int)roundf(fmaxf(0.0f,fminf(255.0,red[idx]  *255.0f/white_value)));
        bint[idx] =    (int)roundf(fmaxf(0.0f,fminf(255.0,blue[idx] *255.0f/white_value)));
    }
}


/* this kernel performs bilateral filtering on the image*/

__forceinline__ __device__ int reflect_padding(int id, int limit)
{
    if(id < 0) return -1 * id;
    else if(id >= limit) return 2*(limit-1) -id;
    else return id;
} 

__global__ void Bilateral_filter_kernel(float* channel_input, float* channel_output, int shared_size, int padding, float dim_variance, float range_variance,  int width, int length)
{
    int x=blockIdx.x * blockDim.x + threadIdx.x, y=blockIdx.y * blockDim.y + threadIdx.y;
    int idx= (y)* width  + (x);  
    //int shared_size = block_dim + 2*padding;
    int ty = threadIdx.y, tx = threadIdx.x;
    int ty0 = ty+padding, tx0 = tx+padding;
    extern __shared__ float buffer[];
    for(int l = 0; (ty + l * block_dim) < shared_size; l++)
        for(int m = 0; (tx + m * block_dim)< shared_size; m++)
        {
            int x0 = reflect_padding((x - padding) + m * block_dim , width );
            int y0 = reflect_padding((y - padding) + l * block_dim , length );

            buffer[(ty + l * block_dim) * shared_size + (tx + m * block_dim)] = channel_input[y0 * width + x0];
        }

    __syncthreads();
    if(y<length && x<width)
    {
        float central_pixel = buffer[ty0 * shared_size + tx0];
        float norm_sum = 0.0f, kernel_sum = 0.0f;
        float dim_pdt = 1.0f / (2.0f *dim_variance*dim_variance);
        float range_pdt = 1.0f / (2.0f *range_variance * range_variance);
        for(int i=-padding; i <= padding; i++)
            for(int j=-padding; j <= padding; j++)
            {
                float neighbor_pixel = buffer[(ty0 + i) * shared_size +(tx0+j)],temp, diff = central_pixel - neighbor_pixel;
                temp = __expf( -(( i*i + j*j )*(dim_pdt) + ((diff )*(diff) )* (range_pdt)));
                norm_sum += temp;
                kernel_sum += neighbor_pixel* temp;
            }
        channel_output[idx] = (kernel_sum / norm_sum);
    }

}

// /* this kernel performs bilateral filtering on the image*/
// __global__ void Guided_filter_kernel(float* channel, int kernal_size, int padding, int width, int length)
// {

//     int x=blockIdx.x * blockDim.x + threadIdx.x, y=blockIdx.y * blockDim.y + threadIdx.y;
//     int idx= (y)* width  + (x);  
//     int shared_size = block_dim + 2*padding;
//     int ty = threadIdx.y, tx = threadIdx.x;

//     extern __shared__ float buffer[];
//     for(int l = 0; (ty + l * block_dim) < shared_size; l++)
//         for(int m = 0; (tx + m * block_dim)< shared_size; m++)
//         {
//             int x0 = reflect_padding((x - padding) + m * block_dim , width );
//             int y0 = reflect_padding((y - padding) + l * block_dim , length );

//             buffer[(ty + l * block_dim) * shared_size + (tx + m * block_dim)] = channel[y0 * width + x0];
//         }

//     __syncthreads();
//     if(y<length && x<width)
//     {

//     }
// }

// /* this kernel performs gaussian filtering on the image*/
// __global__ void Guided_filter_kernel(float* channel, int kernal_size, int padding, int width, int length)
// {
//     int x=blockIdx.x * blockDim.x + threadIdx.x, y=blockIdx.y * blockDim.y + threadIdx.y;
//     int idx= (y)* width  + (x);  
//     int shared_size = block_dim + 2*padding;
//     int ty = threadIdx.y, tx = threadIdx.x;

//     extern __shared__ float buffer[];
//     for(int l = 0; (ty + l * block_dim) < shared_size; l++)
//         for(int m = 0; (tx + m * block_dim)< shared_size; m++)
//         {
//             int x0 = reflect_padding((x - padding) + m * block_dim , width );
//             int y0 = reflect_padding((y - padding) + l * block_dim , length );

//             buffer[(ty + l * block_dim) * shared_size + (tx + m * block_dim)] = channel[y0 * width + x0];
//         }

//     __syncthreads();
//     if(y<length && x<width)
//     {

//     }
// }



// Host code for Image Signal Processing Pipeline
py::tuple ISP(py::array_t<float> Image, const configuration& cfg )    
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
    float *Input = static_cast<float*>(buffer.ptr);
    int array_size = static_cast<int>(buffer.size);

    ////////////////////////////////////////////////////
    //
    //
    // ### creating Device variables for loading data to GPU. D_Image_1 and D_Image_2 are image input output pairs which change job after each kernel. ###
    //
    //
    ////////////////////////////////////////////////////
    float *D_Image_1, *D_image_2; 
    cudaMalloc( &D_Image_1, array_size * sizeof(float)); // creating memory pointers on gpu memory for image.
    cudaMalloc( &D_image_2, array_size * sizeof(float));

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
    cudaMemcpy(D_Image_1 , Input, array_size * sizeof(float), cudaMemcpyHostToDevice);


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
        cudaMemcpy(D_image_2,D_Image_1, array_size * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    ////////////////////////////////////////////////////
    //
    //
    // D_Image_1 is not used after this point therefore the memory is released.
    //
    //
    ////////////////////////////////////////////////////
    float* channel_temp = D_Image_1;

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
        cudaMemcpyToSymbol( D_BLC_Offset, cfg.BLC_Offset.data(), 4 * sizeof(float));

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
            throw std::runtime_error("LSC Gain must contain exactly 4 int values");
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
            double* D_AWBG;
            double H_AWBG[3];

            cudaMalloc( &D_AWBG, 3 * sizeof(double));
            cudaMemset( D_AWBG, 0, 3 * sizeof(double));

            AWBG_kernel<<<dim3(blockx,blocky),dim3(block_dim,block_dim)>>>( D_image_2, D_AWBG, cfg.orientation, cfg.width, cfg.length);
            cudaDeviceSynchronize();

            cudaMemcpy(H_AWBG, D_AWBG, 3 * sizeof(double), cudaMemcpyDeviceToHost);

            GAIN_RED  = (float)((double)(H_AWBG[0])/(double)(2*H_AWBG[2]));
            GAIN_GREEN = 1.0f;
            GAIN_BLUE = (float)((double)(H_AWBG[0])/(double)(2*H_AWBG[1]));
            cudaFree(D_AWBG);

        }
        else
        {
            GAIN_RED = cfg.AWB_gain[0];
            GAIN_GREEN = cfg.AWB_gain[1];
            GAIN_BLUE = cfg.AWB_gain[2];
        }
        AWBG_Apply_kernel<<<dim3(blockx,blocky),dim3(block_dim,block_dim)>>>( D_image_2, GAIN_RED, GAIN_GREEN, GAIN_BLUE, cfg.orientation, cfg.width, cfg.length);
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
    float* CHANNEL_0;
    float* CHANNEL_1;
    float* CHANNEL_2;

    int* RED;
    int* GREEN;
    int* BLUE;

    cudaMalloc(&CHANNEL_0, array_size * sizeof(float));
    cudaMalloc(&CHANNEL_1, array_size * sizeof(float));
    cudaMalloc(&CHANNEL_2, array_size * sizeof(float));

    cudaMalloc(&RED, array_size * sizeof(int));
    cudaMalloc(&GREEN, array_size * sizeof(int));
    cudaMalloc(&BLUE, array_size * sizeof(int));

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
        Transform_Kernel<<<dim3(blockx,blocky),dim3(block_dim,block_dim)>>>(CHANNEL_0, CHANNEL_1, CHANNEL_2, cfg.width, cfg.length);
        cudaDeviceSynchronize();
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
        float CSC[9] = {0.2988, 0.5869, 0.1143, -0.14713, -0.28886, 0.436, 0.615, -0.51499, -0.10001};
        cudaMemcpyToSymbol( D_MAT, CSC, 9 * sizeof(float));
        Transform_Kernel<<<dim3(blockx,blocky),dim3(block_dim,block_dim)>>>(CHANNEL_0, CHANNEL_1, CHANNEL_2, cfg.width, cfg.length);
        cudaDeviceSynchronize();
    }

    ////////////////////////////////////////////////////
    //
    //
    // edge aware low pass filtering - Bilateral kernel on luminance channel
    //  Channel_0 :: Y   Channel_1 :: U    Channel_2 :: V
    //
    ////////////////////////////////////////////////////
    if(cfg.Color_Space_Conversion && cfg.Bilateral_Filter)
    {

        int padding = cfg.Bilateral_kernel_size /2;
        int shared_size = block_dim + 2 * padding;
        float bilateral_scaled_std = ( cfg.Bilateral_Range_STD ) * 65535 / (cfg.white_level);
        size_t shared_memory_vol = shared_size * shared_size * sizeof(float);
        Bilateral_filter_kernel<<<dim3(blockx,blocky),dim3(block_dim,block_dim), shared_memory_vol>>>( CHANNEL_0, channel_temp, shared_size, padding, cfg.Bilateral_Domain_STD , bilateral_scaled_std,  cfg.width, cfg.length);
        cudaDeviceSynchronize();

        float* buf = channel_temp;
        channel_temp = CHANNEL_0;
        CHANNEL_0 = buf;


    }


    ////////////////////////////////////////////////////
    //
    //
    //  Brightness and Saturation Adjustment
    //  Channel_0 :: Y   Channel_1 :: U    Channel_2 :: V
    //
    ////////////////////////////////////////////////////
    if(cfg.Color_Space_Conversion && (cfg.Brightness|| cfg.Saturation))
    {
        if(cfg.Saturation && cfg.Brightness)
        {
            Scaling_kernel_3Channel<<<dim3(blockx,blocky),dim3(block_dim,block_dim)>>>(CHANNEL_0,CHANNEL_1,CHANNEL_2, cfg.Brightness_value, cfg.Saturation_value,  cfg.width, cfg.length);
            cudaDeviceSynchronize();
        }
        else if(cfg.Brightness)
        {
            Scaling_kernel<<<dim3(blockx,blocky),dim3(block_dim,block_dim)>>>(CHANNEL_0, cfg.Brightness_value, cfg.width, cfg.length);
            cudaDeviceSynchronize();
        }
        else if(cfg.Saturation)
        {
            Scaling_kernel_2Channel<<<dim3(blockx,blocky),dim3(block_dim,block_dim)>>>(CHANNEL_1,CHANNEL_2, cfg.Saturation_value,  cfg.width, cfg.length);
            cudaDeviceSynchronize();
        }
        

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
        float CSC[9] = {1,0,1.13983,1,-0.39465,-0.586060,1,2.03211,0};
        cudaMemcpyToSymbol( D_MAT, CSC, 9 * sizeof(float));
        Transform_Kernel<<<dim3(blockx,blocky),dim3(block_dim,block_dim)>>>(CHANNEL_0, CHANNEL_1, CHANNEL_2, cfg.width, cfg.length);
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
        float *D_LUT;
        float x;

        cudaMalloc(&D_LUT, ((int)cfg.white_level+1) * sizeof(float));

        std::vector<float> LUT((int)cfg.white_level+1);

        for(int i=0;i<(cfg.white_level+1);i++)
        {
            x = (float)(i)/(cfg.white_level);
            x = powf(x, (1.0f/cfg.GAMMA_VALUE));

            LUT[i] = (x*cfg.white_level);
        }

        cudaMemcpy(D_LUT, LUT.data() , (cfg.white_level+1) *  sizeof(float), cudaMemcpyHostToDevice);
        LUT_kernel<<<dim3(blockx,blocky),dim3(block_dim,block_dim)>>>( CHANNEL_0, CHANNEL_1, CHANNEL_2, D_LUT, cfg.white_level, cfg.width, cfg.length);
        cudaDeviceSynchronize();
        cudaFree(D_LUT);
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

    Norm_kernel<<<dim3(blockx,blocky),dim3(block_dim,block_dim)>>>( CHANNEL_0, CHANNEL_1, CHANNEL_2, RED, GREEN, BLUE, cfg.white_level, cfg.width, cfg.length);
    cudaDeviceSynchronize();

    cudaMemcpy(R, RED, array_size*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(G, GREEN, array_size*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(B, BLUE, array_size*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(CHANNEL_0);
    cudaFree(CHANNEL_1);
    cudaFree(CHANNEL_2);
    cudaFree(RED);
    cudaFree(GREEN);
    cudaFree(BLUE);
    cudaFree(channel_temp);

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

    LSC_gain : int vector | 4 values
        gain values for leans shading correction in order (00 , 01, 10, 11)
    
    LSC_Max_radius : int
        max value for lens shading correction radius

    AWB : bool
        Enable auto white balance.

    CCM : bool
        Enable color correction matrix.
    
    CCM_gain : int vector : 9 values
        color correction matrix:: as flattened array in order [00 01 02 10 11 12 20 21 22]

    GAMMA : bool
        Enable gamma correction.
    
    GAMMA_VALUE : int
        value for gamma correction.
    
    AWB_gain : int vector
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
        .def_readwrite("Saturation", &configuration::Saturation)
        .def_readwrite("Saturation_value", &configuration::Saturation_value)
        .def_readwrite("AWB_Value_Given", &configuration::AWB_Value_Given)
        .def_readwrite("Color_Space_Conversion", &configuration::Color_Space_Conversion)
        .def_readwrite("Brightness", &configuration::Brightness)
        .def_readwrite("Brightness_value", &configuration::Brightness_value)
        .def_readwrite("Bilateral_Filter", &configuration::Bilateral_Filter)
        .def_readwrite("Bilateral_kernel_size", &configuration::Bilateral_kernel_size)
        .def_readwrite("Bilateral_Domain_STD", &configuration::Bilateral_Domain_STD)
        .def_readwrite("Bilateral_Range_STD", &configuration::Bilateral_Range_STD);
        

        
    // 2. Bind your functions
    // Note: If LSC takes the struct as an argument, define it like this:
    m.def("ISP", &ISP, "Image Signal Processing Pipeline");
}