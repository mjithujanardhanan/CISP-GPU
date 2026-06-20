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

// including required libraries
#include<cuda_runtime.h>
#include<iostream>
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

#define block_dim 16                                                                    // dimension of kernel block size                                                                                                                                                      // Dimension of each cuda block. Each block contains 256 threads representing a pixel each
#define block_size 256                                                                  // total no of threads in a block

namespace py = pybind11;

__constant__ float D_MAT_RGB_YCbCr[9] = {0.2988, 0.5869, 0.1143, -0.1687, -0.3313, 0.5, 0.5, -0.4187, -0.0813};                                                            //Matrix for kernel operations
__constant__ float D_MAT_YCbCr_RGB[9] = {1, -0.0006, 1.4022, 1, -0.34468, -0.7139, 1, 1.77141, 0.00007};
__constant__ float D_MAT[9];
// __constant__ float D_BLC_Offset[4];                                                     //Black level correction 
// __constant__ float D_LSC[4];   
__constant__ int D_LUT[256];                                                         //Lens shading correction values
__constant__ float D_EDGE[256];                                                         //max kernel size = 16 
__constant__ float D_hue[4];                                                            // Rotation matrix for hue adjustment
__constant__ float D_GAUSSIAN[256];                                                     // max Gaussian Kernel size = 16
__constant__ int D_width;                                                               //constant variables for image height and width
__constant__ int D_length;



struct configuration                                                                    //Structure for pipeline configurations
{
    
    int     length=0;                                                                   //Image length
    int     width=0;                                                                    //image width
    float   white_level = 65535.0f;                                                     //assumed 16 bit precision
    int     orientation=0;                                                              //Image orientation

    bool    DPC = false;                                                                //Toogle Defective pixel correction
    float   DPC_threshold=0;                                                            //Defective pixel correction threshold

    bool    BLC = false;                                                                //Black level correction       
    std::vector<float> BLC_Offset;                                                      //Offset for Black level correction

    bool    LSC = false;                                                                //toogle Lens shading correction
    std::vector<float> LSC_gain;                                                        //Lens shading correction gain values
    float   LSC_Max_radius=0.0f;                                                        //Radius of lens vignetting

    bool    AWB = true;                                                                 //toogle automatic white balance
    bool    Exposure = false;                                                           //toogle exposure compensation
    float   Exposure_value = 0.0f;                                                      //Exposure compensation value
    std::vector<float> AWB_gain;                                                        //Automatic white balance gains user defined
    bool    AWB_Value_Given = false;                                                    //toogle AWB if user defined

    std::vector<float> CCM_gain;                                                        //Color correction matix
    bool    CCM=false;                                                                  //toogle Color correction matix

    bool    Color_Space_Conversion = false;                                             //toogle color space conversion. this deactivates entire tonal adjustments and filters
    bool    Brightness = false;                                                         //toogle brightness
    float   Brightness_value = 1.0f;                                                    //scale for brightness
    bool    Saturation = false;                                                         //toogle Saturation
    float   Saturation_value = 1.0f;                                                    //value for saturation
    bool    Hue = false;                                                                //toogle hue
    float   Hue_value = 0.0f;                                                           //in radians from 0 to 2pi (360 degrees)
    bool    Contrast = false;                                                           //toogle contrast
    float   Contrast_value = 1.0f;                                                      //value for contrast
    bool    Tint = false;                                                               //toogle tint
    float   Tint_value = 1.0f;                                                          //value for tint
    bool    Vibrance = false;                                                           //toogle vibrance
    float   Vibrance_value = 0.0f;                                                      //value for vibrance

    bool    Bilateral_Filter = false;                                                   //toogle bilateral filter
    bool    Joint_bilateral_kernel = false;
    int     Bilateral_kernel_size = 3;                                                  //filter kernel size
    float   Bilateral_spatial_STD = 10.0f;                                              // Domain standard deviation
    float   Bilateral_Range_STD = 10.0f;                                                // Range standard deviation

    bool    Edge_enhancement = false;                                                   // Toogle highboost filter
    float   Edge_enhancement_A_Value = 0.0f;                                            // high boost scaling factor
    int     Edge_enhancement_kernel_size = 3;                                           // kernel size
    float   Edge_enhancement_STD = 1;                                                   // edge enhancement standard deviation

    bool    Gaussian_blur = false;
    float   Gaussian_STD = 1.0f;
    int     Gaussian_blur_kernel_size = 3;

    bool    GAMMA=true;                                                                 //Toogle gamma
    float   GAMMA_VALUE = 2.2f;                                                         // value for gamma correction

    
};

__forceinline__ __device__ int reflect_padding(int id, int limit)
{
    if(id < 0) return -1 * id;
    else if(id >= limit) return 2*(limit-1) -id;
    else return id;
} 

struct working_data
{
    float D_BLC_Offset[4];
    float D_LSC[4];
};

__constant__ working_data D_data;

void load_data(configuration cfg)
{

    working_data H_data;

    if(cfg.BLC)
    {
    if(cfg.BLC_Offset.size() != 4)
    {
        throw std::runtime_error("BLC Offset must contain exactly 4 positive integer values");
    }
    memcpy(H_data.D_BLC_Offset, cfg.BLC_Offset.data(), 4 * sizeof(float));
    }
    if(cfg.LSC)
    {
    if(cfg.LSC_gain.size() != 4)
    {
        throw std::runtime_error("LSC Gain must contain exactly 4 values");
    }
    memcpy(H_data.D_LSC, cfg.LSC_gain.data(), 4 * sizeof(float));
    }

    cudaMemcpyToSymbol(D_data, &H_data, 1 * sizeof(working_data));

}


/* this program is an execution of Defective Pixel Consealment on digital bayer domain images*/
__global__ void DP_kernel(float* Image , float* image_out, float threshold)             // Defective pixel correction Image - the image on which the operation is to be performed. Image out- the output image. Threshold- the threshold for dpc correction
{

    // directional gradient calculation
    int x=blockIdx.x * blockDim.x + threadIdx.x, y=blockIdx.y * blockDim.y + threadIdx.y; // y,x are 2d image coordinates calculated from the thread id
    int idx= (y)* D_width  + (x);  // idx is the id of (y,x) element in the flattened image
    if(y<D_length && x<D_width) // out of bounds check
    {

        int up = (y-2)<0?y+2:y-2;               //reflective padding implementation.(important in dpc calculation)
        int down = (y+2)>= D_length ?y-2:y+2;
        int left = (x-2)<0?x+2:x-2;
        int right=(x+2)>= D_width ?x-2:x+2;

        int p1 = up * D_width + x;            //assigning ids to neighbor elements ( future updation to shared memory will remove this part)
        int p2 = up * D_width + left;
        int p3 = y * D_width + left;
        int p4 = down * D_width + left;
        int p5 = down * D_width + x;
        int p6 = down * D_width + right;
        int p7 = y * D_width + right;
        int p8 = up * D_width + right;


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
__global__ void BLC_LSC_kernel(float* Image, bool BLC, bool LSC, float Max_radius)
{
    int x=blockIdx.x * blockDim.x + threadIdx.x, y=blockIdx.y * blockDim.y + threadIdx.y;
    int idx= (y)* D_width  + (x);
    if(y<D_length && x<D_width)
    {
        if(BLC)
            Image[idx] = (Image[idx] - D_data.D_BLC_Offset[(y&1) * 2 + (x&2)]);

        if(LSC)
        {
            int idx= (y)* D_width  + (x);
            float dx = float(D_width)/2.0f  -(float)x; //dx distance from centre to x (here x is x)
            float dy = float(D_length)/2.0f -(float)y;//dy distance from centre to y (here y is y)
            float r = sqrtf(dx*dx + dy*dy); //radius r calculation
            
            Image[idx] = Image[idx]*( 1.0f + r*D_data.D_LSC[(y&1) * 2 + (x&2)]/ Max_radius);  /*lens shading correction modelled as a linear function (original lens shading is modelled
                                                                                    as a cos^4 function which will be implemented in a future version. this is adopted only for development purpose)*/
        }
    }

}


/* this program is an calculation of automatic white balance gain on digital bayer domain images*/
__global__ void AWBG_kernel(float* Image , double* awbg, int orientation)                                                // || BGGR - 0 ||  GBRG -1 || GRBG -2 || RGGB -3 ||
{
    int x=blockIdx.x * blockDim.x + threadIdx.x, y=blockIdx.y * blockDim.y + threadIdx.y;                                                                   // calculating the pixel coordinates y and x corresponding to the thread.

    __shared__ float Green_sum[block_size], Red_sum[block_size], Blue_sum[block_size];                                                                     // Shared memory initialization for each channel   int x=blockIdx.x * blockDim.x + threadIdx.x, y=blockIdx.y * blockDim.y + threadIdx.y;  

    int threadid = threadIdx.y*blockDim.x + threadIdx.x;

    Green_sum[threadid]  = 0;                                                                                                                               //color_sum corresponds to variable to store the sum of each color pixels. 
    Blue_sum[threadid]  = 0;                                                                                                                                //One can assume an overflow issue but for a double to overflow with a 16 bit precision it will take 
    Red_sum[threadid]  = 0;                                                                                                                                 // all the memory ever produced and will produce to store such an image.

    if(y<D_length && x<D_width)
    {
        int idx= (y)* D_width  + (x);    
        bool y_parity = y&1, x_parity = x&1;                                                                                                                       // Idx is the index of the pixel in the flattened array. 
        
            switch (orientation)
            {


                case 0:
                    /* code */
                    if(y_parity && x_parity)
                    {
                        Red_sum[threadid]  =Image[idx];
                    }
                    else if(!y_parity && !x_parity)
                    {
                        Blue_sum[threadid]  =Image[idx];
                    }
                    else
                    {
                        Green_sum[threadid]  = Image[idx];
                    }
                    
                    break;

                case 1:
                    /* code */
                    if(!y_parity && x_parity)
                    {
                        Red_sum[threadid]  =Image[idx];
                    }
                    else if(y_parity && !x_parity)
                    {
                        Blue_sum[threadid]  =Image[idx];
                    }
                    else
                    {
                        Green_sum[threadid]  = Image[idx];
                    }
                    break;

                case 2:
                    /* code */
                    if(!y_parity && x_parity)
                    {
                        Blue_sum[threadid]  =Image[idx];
                    }
                    else if(y_parity && !x_parity)
                    {
                        Red_sum[threadid]  =Image[idx];
                    }
                    else
                    {
                        Green_sum[threadid]  = Image[idx];
                    }
                    break;

                case 3:
                    /* code */
                    if(y_parity && x_parity)
                    {
                        Blue_sum[threadid]  =Image[idx];
                    }
                    else if(!y_parity && !x_parity)
                    {
                        Red_sum[threadid]  =Image[idx];
                    }
                    else
                    {
                        Green_sum[threadid]  = Image[idx];
                    }
                    break;
                
                default:
                    break;
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
    /*In this program each thread identifies and stores a pixel in one of the three colors. then using parallel reduction technique
        accumulates them locally in the block. the thread with id 0 adds the values to the global sum.
    */
}

__global__ void AWBG_Apply_kernel(float* Image ,float gain_r,float gain_g,float gain_b, int orientation)                // || BGGR - 0 ||  GBRG -1 || GRBG -2 || RGGB -3 ||    
{
    int x=blockIdx.x * blockDim.x + threadIdx.x, y=blockIdx.y * blockDim.y + threadIdx.y;
    int idx= (y)* D_width  + (x);


    if(y<D_length && x<D_width)
    {
            if(y&1 && x&1)
            {
                if(orientation ==1 || orientation ==2)
                {
                    Image[idx] = Image[idx] * gain_g ;
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
    /* Kernel for gain application.
    */

}

__global__ void AWBG_Apply_kernel(float* Image ,float gain_r,float gain_g,float gain_b, float Egain, int orientation)                // || BGGR - 0 ||  GBRG -1 || GRBG -2 || RGGB -3 ||    
{
    int x=blockIdx.x * blockDim.x + threadIdx.x, y=blockIdx.y * blockDim.y + threadIdx.y;
    int idx= (y)* D_width  + (x);


    if(y<D_length && x<D_width)
    {
            if(y&1 && x&1)
            {
                if(orientation ==1 || orientation ==2)
                {
                    Image[idx] = Image[idx] * gain_g * Egain;
                }
                else if(orientation ==0)
                {
                    Image[idx] = Image[idx] * gain_r* Egain;
                }
                else
                {
                    Image[idx] = Image[idx] * gain_b* Egain;
                }
            }
            else if(!(y&1) && !(x&1))
            {
                if(orientation ==1 || orientation ==2)
                {
                    Image[idx] = Image[idx] * gain_g* Egain;
                }
                else if(orientation ==0)
                {
                    Image[idx] = Image[idx] * gain_b* Egain;
                }
                else
                {
                    Image[idx] = Image[idx] * gain_r* Egain;
                }
            }
        
            else if((y&1) && !(x&1))
            {
                if(orientation ==0 || orientation ==3)
                {
                    Image[idx] = Image[idx] * gain_g* Egain;
                }
                else if(orientation ==1)
                {
                    Image[idx] = Image[idx] * gain_b* Egain;
                }
                else
                {
                    Image[idx] = Image[idx] * gain_r* Egain;
                }
            }
            else 
            {
                if(orientation ==0 || orientation ==3)
                {
                    Image[idx] = Image[idx] * gain_g* Egain;
                }
                else if(orientation ==1)
                {
                    Image[idx] = Image[idx] * gain_r* Egain;
                }
                else
                {
                    Image[idx] = Image[idx] * gain_b* Egain;
                }
            }
    }
    /* Kernel for gain application.
    */

}

/* this program is an execution of edge aware interpolation of bayer domain image*/
__global__ void DEBAYER_kernel_1(float* Image , float* output, int shared_size, int padding, int orientation)                                         //green interpolation (this is a trial to test if cooperative loading using shared memory or using threads for each halo is faster. the kernel will be replaced with most efficient one after profiling. )
{
    int x=blockIdx.x * blockDim.x + threadIdx.x, y=blockIdx.y * blockDim.y + threadIdx.y;
    int idx= (y)* D_width  + (x);  
    //int shared_size = block_dim + 2*padding;
    int ty = threadIdx.y, tx = threadIdx.x;
    int ty0 = ty+padding, tx0 = tx+padding;
    extern __shared__ float buffer[];
    for(int l = 0; (ty + l * block_dim) < shared_size; l++)
        for(int m = 0; (tx + m * block_dim)< shared_size; m++)
        {
            int x0 = reflect_padding((x - padding) + m * block_dim , D_width );
            int y0 = reflect_padding((y - padding) + l * block_dim , D_length );

            buffer[(ty + l * block_dim) * shared_size + (tx + m * block_dim)] = Image[y0 * D_width + x0];
        }

    __syncthreads();


    if(y<D_length && x<D_width)
    {
        if(orientation == 0 || orientation == 3)
        {
            if( (x+y) & 1 )
            {
                output[idx] = buffer[ty0 * shared_size + tx0 ];
            }
            else
            {
                float dv = fabsf(buffer[(ty0-1) * shared_size + tx0] - buffer[(ty0+1) * shared_size + tx0]) + fabsf(2* buffer[ty0 * shared_size + tx0] -(buffer[(ty0-2) * shared_size + tx0] + buffer[(ty0+2)* shared_size +  tx0]));
                float dh = fabsf(buffer[ty0 * shared_size + (tx0-1)] - buffer[ty0 * shared_size + (tx0+1)]) + fabsf(2* buffer[ty0 * shared_size + tx0] -(buffer[ty0 * shared_size + (tx0-2)] + buffer[ty0 * shared_size + (tx0+2)]));


                if (dh>dv)
                {
                    output[idx] = ((buffer[(ty0-1) * shared_size + tx0] + buffer[(ty0+1) * shared_size + tx0])*0.5 + (2* buffer[ty0 * shared_size + tx0] -(buffer[(ty0-2) * shared_size + tx0] + buffer[(ty0+2)* shared_size +  tx0]))*0.25f);
                }
                
                else if (dh<dv)
                {
                    output[idx] = ((buffer[ty0 * shared_size + (tx0-1)] + buffer[ty0 * shared_size + (tx0+1)])*0.5 + (2* buffer[ty0 * shared_size + tx0] -(buffer[ty0 * shared_size + (tx0-2)] + buffer[ty0 * shared_size + (tx0+2)]))*0.25f);
                }
                else
                {
                    output[idx] = (((buffer[(ty0-1) * shared_size + tx0] + buffer[(ty0+1) * shared_size + tx0] + buffer[ty0 * shared_size + (tx0-1)] + buffer[ty0 * shared_size + (tx0+1)])*0.25 + (2* buffer[ty0 * shared_size + tx0] -(buffer[(ty0-2) * shared_size + tx0] + buffer[(ty0+2)* shared_size +  tx0]) +2* buffer[ty0 * shared_size + tx0] -(buffer[ty0 * shared_size + (tx0-2)] + buffer[ty0 * shared_size + (tx0+2)]))*0.125f));
                }
                
            }

        }

        else if(orientation == 1 || orientation == 2)
        {
            if(!((x+y)&1))
            {
                output[idx]=buffer[ty0 * shared_size + tx0 ];
            }
            else
            {
                float dv = fabsf(buffer[(ty0-1) * shared_size + tx0] - buffer[(ty0+1) * shared_size + tx0]) + fabsf(2* buffer[ty0 * shared_size + tx0] -(buffer[(ty0-2) * shared_size + tx0] + buffer[(ty0+2)* shared_size +  tx0]));
                float dh = fabsf(buffer[ty0 * shared_size + (tx0-1)] - buffer[ty0 * shared_size + (tx0+1)]) + fabsf(2* buffer[ty0 * shared_size + tx0] -(buffer[ty0 * shared_size + (tx0-2)] + buffer[ty0 * shared_size + (tx0+2)]));

                if (dh>dv)
                {
                    output[idx] = ((buffer[(ty0-1) * shared_size + tx0] + buffer[(ty0+1) * shared_size + tx0])*0.5 + (2* buffer[ty0 * shared_size + tx0] -(buffer[(ty0-2) * shared_size + tx0] + buffer[(ty0+2)* shared_size +  tx0]))*0.25f);
                }
                
                else if (dh<dv)
                {
                    output[idx] = ((buffer[ty0 * shared_size + (tx0-1)] + buffer[ty0 * shared_size + (tx0+1)])*0.5 + (2* buffer[ty0 * shared_size + tx0] -(buffer[ty0 * shared_size + (tx0-2)] + buffer[ty0 * shared_size + (tx0+2)]))*0.25f);
                }
                else
                {
                    output[idx] = (((buffer[(ty0-1) * shared_size + tx0] + buffer[(ty0+1) * shared_size + tx0] + buffer[ty0 * shared_size + (tx0-1)] + buffer[ty0 * shared_size + (tx0+1)])*0.25 + (2* buffer[ty0 * shared_size + tx0] -(buffer[(ty0-2) * shared_size + tx0] + buffer[(ty0+2)* shared_size +  tx0]) +2* buffer[ty0 * shared_size + tx0] -(buffer[ty0 * shared_size + (tx0-2)] + buffer[ty0 * shared_size + (tx0+2)]))*0.125f));
                }


            }

        }
        
    }

}

__global__ void DEBAYER_kernel_2(float* Image , float* green, float* red, float* blue, int shared_size,int shared_vol, int padding, int orientation)                 //color interpolation
{
    int x=blockIdx.x * blockDim.x + threadIdx.x, y=blockIdx.y * blockDim.y + threadIdx.y;
    int idx= (y)* D_width  + (x);  
    //int shared_size = block_dim + 2*padding;
    int ty = threadIdx.y, tx = threadIdx.x;
    int ty0 = ty+padding, tx0 = tx+padding;
    extern __shared__ float buffer[];
    for(int l = 0; (ty + l * block_dim) < shared_size; l++)
        for(int m = 0; (tx + m * block_dim)< shared_size; m++)
        {
            int x0 = reflect_padding((x - padding) + m * block_dim , D_width );
            int y0 = reflect_padding((y - padding) + l * block_dim , D_length );

            buffer[(ty + l * block_dim) * shared_size + (tx + m * block_dim)] = Image[y0 * D_width + x0];
            buffer[shared_vol +  (ty + l * block_dim) * shared_size + (tx + m * block_dim)] = green[y0 * D_width + x0];
        }

    __syncthreads();

    if(y<D_length && x<D_width)
    {
        bool x_parity = x&1, y_parity = y&1;
        switch (orientation)
        {

            case 0:
                
                    
                if(!x_parity && !y_parity)
                {
                    blue[idx] =(buffer[ty0 * shared_size + tx0]);
                    red[idx]  =(buffer[shared_vol + ty0 * shared_size + tx0]+0.25f*(buffer[(ty0-1) * shared_size + (tx0-1)]-buffer[shared_vol + (ty0-1) * shared_size + (tx0-1)] + buffer[(ty0+1) * shared_size + (tx0+1)]-buffer[shared_vol + (ty0+1) * shared_size + (tx0+1)] +buffer[(ty0-1) * shared_size + (tx0+1)]-buffer[shared_vol + (ty0-1) * shared_size + (tx0+1)] + buffer[(ty0+1) * shared_size + (tx0-1)]-buffer[shared_vol + (ty0+1) * shared_size + (tx0-1)]));
                }
                else if(x_parity && !y_parity)
                {
                    red[idx] = (buffer[shared_vol + ty0 * shared_size + tx0]+0.5f *(buffer[(ty0-1) * shared_size + tx0]-buffer[shared_vol + (ty0-1) * shared_size + tx0] + (buffer[(ty0+1) * shared_size + tx0]-buffer[shared_vol + (ty0+1) * shared_size + tx0])));                     //vertical interpolation
                    blue[idx]= (buffer[shared_vol + ty0 * shared_size + tx0]+0.5f *(buffer[ty0 * shared_size + (tx0-1)]-buffer[shared_vol + ty0 * shared_size + (tx0-1)] + (buffer[ty0 * shared_size + (tx0+1)]-buffer[shared_vol + ty0 * shared_size + (tx0+1)])));                     //horizontal interpolation
                }
                else if(!x_parity && y_parity)
                {
                    blue[idx] = (buffer[shared_vol + ty0 * shared_size + tx0]+0.5f *(buffer[(ty0-1) * shared_size + tx0]-buffer[shared_vol + (ty0-1) * shared_size + tx0] + (buffer[(ty0+1) * shared_size + tx0]-buffer[shared_vol + (ty0+1) * shared_size + tx0])));                     //vertical interpolation
                    red[idx]  = (buffer[shared_vol + ty0 * shared_size + tx0]+0.5f *(buffer[ty0 * shared_size + (tx0-1)]-buffer[shared_vol + ty0 * shared_size + (tx0-1)] + (buffer[ty0 * shared_size + (tx0+1)]-buffer[shared_vol + ty0 * shared_size + (tx0+1)])));                     //horizontal interpolation
                }
                else
                {
                    red[idx] =(buffer[ty0 * shared_size + tx0]);
                    blue[idx]=(buffer[shared_vol + ty0 * shared_size + tx0]+0.25f*(buffer[(ty0-1) * shared_size + (tx0-1)]-buffer[shared_vol + (ty0-1) * shared_size + (tx0-1)] + buffer[(ty0+1) * shared_size + (tx0+1)]-buffer[shared_vol + (ty0+1) * shared_size + (tx0+1)] +buffer[(ty0-1) * shared_size + (tx0+1)]-buffer[shared_vol + (ty0-1) * shared_size + (tx0+1)] + buffer[(ty0+1) * shared_size + (tx0-1)]-buffer[shared_vol + (ty0+1) * shared_size + (tx0-1)]));

                }

                


                break;
            

            case 1:
                
                if(!x_parity && !y_parity)
                {
                    red[idx] = (buffer[shared_vol + ty0 * shared_size + tx0]+0.5f *(buffer[(ty0-1) * shared_size + tx0]-buffer[shared_vol + (ty0-1) * shared_size + tx0] + (buffer[(ty0+1) * shared_size + tx0]-buffer[shared_vol + (ty0+1) * shared_size + tx0])));                     //vertical interpolation
                    blue[idx]= (buffer[shared_vol + ty0 * shared_size + tx0]+0.5f *(buffer[ty0 * shared_size + (tx0-1)]-buffer[shared_vol + ty0 * shared_size + (tx0-1)] + (buffer[ty0 * shared_size + (tx0+1)]-buffer[shared_vol + ty0 * shared_size + (tx0+1)])));                     //horizontal interpolation

                }
                else if(x_parity && !y_parity)
                {
                    blue[idx] = (buffer[ty0 * shared_size + tx0]);
                    red[idx]  = (buffer[shared_vol + ty0 * shared_size + tx0]+0.25f*(buffer[(ty0-1) * shared_size + (tx0-1)]-buffer[shared_vol + (ty0-1) * shared_size + (tx0-1)] + buffer[(ty0+1) * shared_size + (tx0+1)]-buffer[shared_vol + (ty0+1) * shared_size + (tx0+1)] +buffer[(ty0-1) * shared_size + (tx0+1)]-buffer[shared_vol + (ty0-1) * shared_size + (tx0+1)] + buffer[(ty0+1) * shared_size + (tx0-1)]-buffer[shared_vol + (ty0+1) * shared_size + (tx0-1)]));
                }
                else if(!x_parity && y_parity)
                {
                    red[idx] = (buffer[ty0 * shared_size + tx0]);
                    blue[idx]= (buffer[shared_vol + ty0 * shared_size + tx0]+0.25f*(buffer[(ty0-1) * shared_size + (tx0-1)]-buffer[shared_vol + (ty0-1) * shared_size + (tx0-1)] + buffer[(ty0+1) * shared_size + (tx0+1)]-buffer[shared_vol + (ty0+1) * shared_size + (tx0+1)] +buffer[(ty0-1) * shared_size + (tx0+1)]-buffer[shared_vol + (ty0-1) * shared_size + (tx0+1)] + buffer[(ty0+1) * shared_size + (tx0-1)]-buffer[shared_vol + (ty0+1) * shared_size + (tx0-1)]));
                }
                else
                {
                    blue[idx] = (buffer[shared_vol + ty0 * shared_size + tx0]+0.5f *(buffer[(ty0-1) * shared_size + tx0]-buffer[shared_vol + (ty0-1) * shared_size + tx0] + (buffer[(ty0+1) * shared_size + tx0]-buffer[shared_vol + (ty0+1) * shared_size + tx0])));                     //vertical interpolation
                    red[idx]  = (buffer[shared_vol + ty0 * shared_size + tx0]+0.5f *(buffer[ty0 * shared_size + (tx0-1)]-buffer[shared_vol + ty0 * shared_size + (tx0-1)] + (buffer[ty0 * shared_size + (tx0+1)]-buffer[shared_vol + ty0 * shared_size + (tx0+1)])));                     //horizontal interpolation
                }

                
                break;


            case 2:
                
                if(!x_parity && !y_parity)
                {
                    blue[idx] = (buffer[shared_vol + ty0 * shared_size + tx0]+0.5f *(buffer[(ty0-1) * shared_size + tx0]-buffer[shared_vol + (ty0-1) * shared_size + tx0] + (buffer[(ty0+1) * shared_size + tx0]-buffer[shared_vol + (ty0+1) * shared_size + tx0])));                     //vertical interpolation
                    red[idx]  = (buffer[shared_vol + ty0 * shared_size + tx0]+0.5f *(buffer[ty0 * shared_size + (tx0-1)]-buffer[shared_vol + ty0 * shared_size + (tx0-1)] + (buffer[ty0 * shared_size + (tx0+1)]-buffer[shared_vol + ty0 * shared_size + (tx0+1)])));                     //horizontal interpolation
                }
                else if(x_parity && !y_parity)
                {
                    red[idx] =(buffer[ty0 * shared_size + tx0]);
                    blue[idx]=(buffer[shared_vol + ty0 * shared_size + tx0]+0.25f*(buffer[(ty0-1) * shared_size + (tx0-1)]-buffer[shared_vol + (ty0-1) * shared_size + (tx0-1)] + buffer[(ty0+1) * shared_size + (tx0+1)]-buffer[shared_vol + (ty0+1) * shared_size + (tx0+1)] +buffer[(ty0-1) * shared_size + (tx0+1)]-buffer[shared_vol + (ty0-1) * shared_size + (tx0+1)] + buffer[(ty0+1) * shared_size + (tx0-1)]-buffer[shared_vol + (ty0+1) * shared_size + (tx0-1)]));
                }
                else if(!x_parity && y_parity)
                {
                    blue[idx] =(buffer[ty0 * shared_size + tx0]);
                    red[idx]  =(buffer[shared_vol + ty0 * shared_size + tx0]+0.25f*(buffer[(ty0-1) * shared_size + (tx0-1)]-buffer[shared_vol + (ty0-1) * shared_size + (tx0-1)] + buffer[(ty0+1) * shared_size + (tx0+1)]-buffer[shared_vol + (ty0+1) * shared_size + (tx0+1)] +buffer[(ty0-1) * shared_size + (tx0+1)]-buffer[shared_vol + (ty0-1) * shared_size + (tx0+1)] + buffer[(ty0+1) * shared_size + (tx0-1)]-buffer[shared_vol + (ty0+1) * shared_size + (tx0-1)]));
                }
                else
                {
                    red[idx] = (buffer[shared_vol + ty0 * shared_size + tx0]+0.5f *(buffer[(ty0-1) * shared_size + tx0]-buffer[shared_vol + (ty0-1) * shared_size + tx0] + (buffer[(ty0+1) * shared_size + tx0]-buffer[shared_vol + (ty0+1) * shared_size + tx0])));                     //vertical interpolation
                    blue[idx]= (buffer[shared_vol + ty0 * shared_size + tx0]+0.5f *(buffer[ty0 * shared_size + (tx0-1)]-buffer[shared_vol + ty0 * shared_size + (tx0-1)] + (buffer[ty0 * shared_size + (tx0+1)]-buffer[shared_vol + ty0 * shared_size + (tx0+1)])));                     //horizontal interpolation
                    
                }

                
                break;


            case 3:
                
                if(!x_parity && !y_parity)
                {
                    red[idx] = (buffer[ty0 * shared_size + tx0]);
                    blue[idx]= (buffer[shared_vol + ty0 * shared_size + tx0]+0.25f*(buffer[(ty0-1) * shared_size + (tx0-1)]-buffer[shared_vol + (ty0-1) * shared_size + (tx0-1)] + buffer[(ty0+1) * shared_size + (tx0+1)]-buffer[shared_vol + (ty0+1) * shared_size + (tx0+1)] +buffer[(ty0-1) * shared_size + (tx0+1)]-buffer[shared_vol + (ty0-1) * shared_size + (tx0+1)] + buffer[(ty0+1) * shared_size + (tx0-1)]-buffer[shared_vol + (ty0+1) * shared_size + (tx0-1)]));
                }
                else if(x_parity && !y_parity)
                {
                    blue[idx] = (buffer[shared_vol + ty0 * shared_size + tx0]+0.5f *(buffer[(ty0-1) * shared_size + tx0]-buffer[shared_vol + (ty0-1) * shared_size + tx0] + (buffer[(ty0+1) * shared_size + tx0]-buffer[shared_vol + (ty0+1) * shared_size + tx0])));                     //vertical interpolation
                    red[idx]  = (buffer[shared_vol + ty0 * shared_size + tx0]+0.5f *(buffer[ty0 * shared_size + (tx0-1)]-buffer[shared_vol + ty0 * shared_size + (tx0-1)] + (buffer[ty0 * shared_size + (tx0+1)]-buffer[shared_vol + ty0 * shared_size + (tx0+1)])));                     //horizontal interpolation

                }
                else if(!x_parity && y_parity)
                {
                    red[idx] = (buffer[shared_vol + ty0 * shared_size + tx0]+0.5f *(buffer[(ty0-1) * shared_size + tx0]-buffer[shared_vol + (ty0-1) * shared_size + tx0] + (buffer[(ty0+1) * shared_size + tx0]-buffer[shared_vol + (ty0+1) * shared_size + tx0])));                     //vertical interpolation
                    blue[idx]= (buffer[shared_vol + ty0 * shared_size + tx0]+0.5f *(buffer[ty0 * shared_size + (tx0-1)]-buffer[shared_vol + ty0 * shared_size + (tx0-1)] + (buffer[ty0 * shared_size + (tx0+1)]-buffer[shared_vol + ty0 * shared_size + (tx0+1)])));                     //horizontal interpolation
                }
                else
                {
                    blue[idx] = (buffer[ty0 * shared_size + tx0]);
                    red[idx]  = (buffer[shared_vol + ty0 * shared_size + tx0]+0.25f*(buffer[(ty0-1) * shared_size + (tx0-1)]-buffer[shared_vol + (ty0-1) * shared_size + (tx0-1)] + buffer[(ty0+1) * shared_size + (tx0+1)]-buffer[shared_vol + (ty0+1) * shared_size + (tx0+1)] +buffer[(ty0-1) * shared_size + (tx0+1)]-buffer[shared_vol + (ty0-1) * shared_size + (tx0+1)] + buffer[(ty0+1) * shared_size + (tx0-1)]-buffer[shared_vol + (ty0+1) * shared_size + (tx0-1)]));
                }

                
                break;


            default:
                
                break;
        }

    }

}


/* this program is for applying transform matrix to the color image*/
__global__ void Transform_Kernel_RGB_YCbCr(float* channel_1, float* channel_2, float* channel_3)
{
    int x=blockIdx.x * blockDim.x + threadIdx.x, y=blockIdx.y * blockDim.y + threadIdx.y;
    
    
    if(y<D_length && x<D_width)
    {
        int idx= (y)* D_width  + (x); 

        float Temp_c1 =    channel_1[idx];
        float Temp_c2 =    channel_2[idx];
        float Temp_c3 =    channel_3[idx];

        channel_1[idx] =     (D_MAT_RGB_YCbCr[0] * Temp_c1 + D_MAT_RGB_YCbCr[1] * Temp_c2 + D_MAT_RGB_YCbCr[2] * Temp_c3);
        channel_2[idx]=      (D_MAT_RGB_YCbCr[3] * Temp_c1 + D_MAT_RGB_YCbCr[4] * Temp_c2 + D_MAT_RGB_YCbCr[5] * Temp_c3);
        channel_3[idx]=      (D_MAT_RGB_YCbCr[6] * Temp_c1 + D_MAT_RGB_YCbCr[7] * Temp_c2 + D_MAT_RGB_YCbCr[8] * Temp_c3);


    }
}

__global__ void Transform_Kernel_YCbCr_RGB(float* channel_1, float* channel_2, float* channel_3)
{
    int x=blockIdx.x * blockDim.x + threadIdx.x, y=blockIdx.y * blockDim.y + threadIdx.y;
    
    
    if(y<D_length && x<D_width)
    {
        int idx= (y)* D_width  + (x); 

        float Temp_c1 =    channel_1[idx];
        float Temp_c2 =    channel_2[idx];
        float Temp_c3 =    channel_3[idx];

        channel_1[idx] =     (D_MAT_YCbCr_RGB[0] * Temp_c1 + D_MAT_YCbCr_RGB[1] * Temp_c2 + D_MAT_YCbCr_RGB[2] * Temp_c3);
        channel_2[idx]=      (D_MAT_YCbCr_RGB[3] * Temp_c1 + D_MAT_YCbCr_RGB[4] * Temp_c2 + D_MAT_YCbCr_RGB[5] * Temp_c3);
        channel_3[idx]=      (D_MAT_YCbCr_RGB[6] * Temp_c1 + D_MAT_YCbCr_RGB[7] * Temp_c2 + D_MAT_YCbCr_RGB[8] * Temp_c3);


    }
}

__global__ void Transform_Kernel(float* channel_1, float* channel_2, float* channel_3)
{
    int x=blockIdx.x * blockDim.x + threadIdx.x, y=blockIdx.y * blockDim.y + threadIdx.y;
    
    
    if(y<D_length && x<D_width)
    {
        int idx= (y)* D_width  + (x); 

        float Temp_c1 =    channel_1[idx];
        float Temp_c2 =    channel_2[idx];
        float Temp_c3 =    channel_3[idx];

        channel_1[idx] =     (D_MAT[0] * Temp_c1 + D_MAT[1] * Temp_c2 + D_MAT[2] * Temp_c3);
        channel_2[idx]=      (D_MAT[3] * Temp_c1 + D_MAT[4] * Temp_c2 + D_MAT[5] * Temp_c3);
        channel_3[idx]=      (D_MAT[6] * Temp_c1 + D_MAT[7] * Temp_c2 + D_MAT[8] * Temp_c3);


    }
}

/* This program is for applying Mapping using a Look Up Table(LUT) to the image on all channels*/
__global__ void LUT_kernel_Gamma( float* red, float* green, float* blue, int* rint, int* gint, int*bint, float i_white_value)
{
    int x=blockIdx.x * block_dim + threadIdx.x, y=blockIdx.y * block_dim + threadIdx.y;
    int idx= (y)* D_width  + (x);  
    if(y<D_length && x<D_width)
    {

        float g_pixel = green[idx];
        float r_pixel = red[idx];
        float b_pixel = blue[idx];

        g_pixel = roundf(fmaxf(0.0f,fminf(255.0, g_pixel * i_white_value)));
        r_pixel = roundf(fmaxf(0.0f,fminf(255.0, r_pixel * i_white_value)));
        b_pixel = roundf(fmaxf(0.0f,fminf(255.0, b_pixel * i_white_value)));


        gint[idx] = D_LUT[(int)g_pixel];
        rint[idx] = D_LUT[(int)r_pixel];
        bint[idx] = D_LUT[(int)b_pixel];
    }
}

/* This program is for applying Mapping using a Look Up Table(LUT) to the image on all channels*/
__global__ void LUT_kernel( float* red, float* green, float* blue, float i_white_value)
{
    int x=blockIdx.x * block_dim + threadIdx.x, y=blockIdx.y * block_dim + threadIdx.y;
    int idx= (y)* D_width  + (x);  
    if(y<D_length && x<D_width)
    {

        float g_pixel = green[idx];
        float r_pixel = red[idx];
        float b_pixel = blue[idx];

        g_pixel = roundf(fmaxf(0.0f,fminf(255.0, g_pixel * i_white_value)));
        r_pixel = roundf(fmaxf(0.0f,fminf(255.0, r_pixel * i_white_value)));
        b_pixel = roundf(fmaxf(0.0f,fminf(255.0, b_pixel * i_white_value)));


        green[idx] = D_LUT[(int)g_pixel];
        red[idx] = D_LUT[(int)r_pixel];
        blue[idx] = D_LUT[(int)b_pixel];
    }
}

/* This program is for applying Mapping using a Look Up Table(LUT) to the image on one channel*/
__global__ void LUT_kernel( float* Channel, float* LUT, float white_value)
{
    int x=blockIdx.x * block_dim + threadIdx.x, y=blockIdx.y * block_dim + threadIdx.y;
    int idx= (y)* D_width  + (x);  
    if(y<D_length && x<D_width)
    {
        Channel[idx] = LUT[(int)roundf(fmaxf(0.0f,fminf(white_value,(Channel[idx]))))];
    }
}

/* This program is for Color control of the image*/
__global__ void Color_control_kernel( float* channel_1, float* channel_2, float* channel_3, float bvalue, float svalue ,float cvalue,float tvalue,float vvalue ,bool brightness,bool saturation,bool hue,bool contrast,bool tint,bool vibrance,float hvalue)
{
    int x=blockIdx.x * block_dim + threadIdx.x, y=blockIdx.y * block_dim + threadIdx.y;
    int idx= (y)* D_width  + (x);  
    if(y<D_length && x<D_width)
    {

        float c1 = channel_1[idx], c2 = channel_2[idx], c3 = channel_3[idx];
        if(brightness)
        {
            c1 += bvalue;
        }
        if(saturation)
        {
            c2 = (c2 * svalue);
            c3 = (c3 * svalue);
        }
        if(hue)
        {
            float temp = c2;
            c2 = D_hue[0] * c2 + D_hue[1] * c3;
            c3 = D_hue[2] * temp + D_hue[3] * c3;
        }
        if(contrast)
        {
            c1 = (c1- hvalue)* cvalue + hvalue;
        }
        if(tint)
        {
            c3 += tvalue;
            c2 -= 0.5f *tvalue;
        }
        if(vibrance)
        {
            float chroma = sqrt(c2*c2 + c3*c3);
            float ch_lim = hvalue * 0.596f ;
            float sat = fminf(chroma / ch_lim, hvalue);

            float gain = 1.0f + vvalue * (1.0f - sat);

            c2 = gain * c2;
            c3 = gain * c3;
        }
        channel_1[idx] = c1;
        channel_2[idx] = c2;
        channel_3[idx] = c3;
    
    }
}

/* This program is for converting datatype of image to 8-bit integer*/
__global__ void Norm_kernel( float* red, float* green, float* blue,int* rint, int* gint, int*bint, float i_white_value)
{
    int x=blockIdx.x * block_dim + threadIdx.x, y=blockIdx.y * block_dim + threadIdx.y;
    int idx= (y)* D_width  + (x);  
    if(y<D_length && x<D_width)
    {
        gint[idx] =    (int)roundf(fmaxf(0.0f,fminf(255.0,green[idx]*i_white_value)));
        rint[idx] =    (int)roundf(fmaxf(0.0f,fminf(255.0,red[idx]  *i_white_value)));
        bint[idx] =    (int)roundf(fmaxf(0.0f,fminf(255.0,blue[idx] *i_white_value)));
    }
}

/* this kernel performs bilateral filtering on the image*/

__global__ void Bilateral_filter_kernel(float* channel_input, float* channel_output, int shared_size, int padding, float spatial_pdt, float range_pdt)
{
    int x=blockIdx.x * blockDim.x + threadIdx.x, y=blockIdx.y * blockDim.y + threadIdx.y;
    int idx= (y)* D_width  + (x);  
    //int shared_size = block_dim + 2*padding;
    int ty = threadIdx.y, tx = threadIdx.x;
    int ty0 = ty+padding, tx0 = tx+padding;
    extern __shared__ float buffer[];
    for(int l = 0; (ty + l * block_dim) < shared_size; l++)
        for(int m = 0; (tx + m * block_dim)< shared_size; m++)
        {
            int x0 = reflect_padding((x - padding) + m * block_dim , D_width );
            int y0 = reflect_padding((y - padding) + l * block_dim , D_length );

            buffer[(ty + l * block_dim) * shared_size + (tx + m * block_dim)] = channel_input[y0 * D_width + x0];
        }

    __syncthreads();
    if(y<D_length && x<D_width)
    {
        float central_pixel = buffer[ty0 * shared_size + tx0];
        float norm_sum = 0.0f, kernel_sum = 0.0f;        
        for(int i=-padding; i <= padding; i++)
            for(int j=-padding; j <= padding; j++)
            {
                float neighbor_pixel = buffer[(ty0 + i) * shared_size +(tx0+j)],temp, diff = central_pixel - neighbor_pixel;

                temp = __expf( -(( i*i + j*j )*(spatial_pdt) + ((diff )*(diff) )* (range_pdt)));

                norm_sum += temp;
                kernel_sum += neighbor_pixel* temp;
            }
        channel_output[idx] = (kernel_sum / norm_sum);
    }

}

__global__ void Bilateral_filter_kernel(float* channel_0,float* channel_1,float* channel_2, float* channel_output_0, float* channel_output_1, float* channel_output_2, int shared_size,  int shared_vol, int padding, float spatial_pdt, float range_pdt)
{
    int x=blockIdx.x * blockDim.x + threadIdx.x, y=blockIdx.y * blockDim.y + threadIdx.y;
    int idx= (y)* D_width  + (x);  
    //int shared_size = block_dim + 2*padding;
    int ty = threadIdx.y, tx = threadIdx.x;
    int ty0 = ty+padding, tx0 = tx+padding;
    extern __shared__ float buffer[];
    for(int l = 0; (ty + l * block_dim) < shared_size; l++)
        for(int m = 0; (tx + m * block_dim)< shared_size; m++)
        {
            int x0 = reflect_padding((x - padding) + m * block_dim , D_width );
            int y0 = reflect_padding((y - padding) + l * block_dim , D_length );

            int sid = ((ty + l * block_dim) * shared_size + (tx + m * block_dim)) * 3;

            buffer[sid] = channel_0[y0 * D_width + x0];
            buffer[sid + 1] = channel_1[y0 * D_width + x0];
            buffer[sid + 2] = channel_2[y0 * D_width + x0];

        }

    __syncthreads();
    if(y<D_length && x<D_width)
    {
        float central_pixel = buffer[(ty0 * shared_size + tx0)*3];
        float norm_sum = 0.0f, kernel_sum_0 = 0.0f, kernel_sum_1 = 0.0f, kernel_sum_2 = 0.0f;
        for(int i=-padding; i <= padding; i++)
            for(int j=-padding; j <= padding; j++)
            {
                int tsid = ((ty0 + i) * shared_size +(tx0+j))* 3;

                float neighbor_pixel_0 = buffer[ tsid],temp;
                float neighbor_pixel_1 = buffer[ tsid +  1];
                float neighbor_pixel_2 = buffer[ tsid  + 2];

                float diff = central_pixel - neighbor_pixel_0;
                temp = __expf( -(( i*i + j*j )*(spatial_pdt) + ((diff )*(diff) )* (range_pdt)));
                norm_sum += temp;
                
                kernel_sum_0 += neighbor_pixel_0* temp;
                kernel_sum_1 += neighbor_pixel_1* temp;
                kernel_sum_2 += neighbor_pixel_2* temp;
            }
        channel_output_0[idx] = (kernel_sum_0 / norm_sum);
        channel_output_1[idx] = (kernel_sum_1 / norm_sum);
        channel_output_2[idx] = (kernel_sum_2 / norm_sum);
    }

}

__global__ void Edge_enhancement_kernel(float* channel_input, float* channel_output, int shared_size, int padding, int kernel_size, float alpha)
{
    int x=blockIdx.x * blockDim.x + threadIdx.x, y=blockIdx.y * blockDim.y + threadIdx.y;
    int idx= (y)* D_width  + (x);  
    //int shared_size = block_dim + 2*padding;
    
    int ty = threadIdx.y, tx = threadIdx.x;
    int ty0 = ty+padding, tx0 = tx+padding;
    extern __shared__ float buffer[];
    for(int l = 0; (ty + l * block_dim) < shared_size; l++)
        for(int m = 0; (tx + m * block_dim)< shared_size; m++)
        {
            int x0 = reflect_padding((x - padding) + m * block_dim , D_width );
            int y0 = reflect_padding((y - padding) + l * block_dim , D_length );

            buffer[(ty + l * block_dim) * shared_size + (tx + m * block_dim)] = channel_input[y0 * D_width + x0];
        }

    
    __syncthreads();
    if(y<D_length && x<D_width)
    {
        float kernel_sum = 0.0f;
        float central_pixel = buffer[ty0 * shared_size + tx0];
        for(int i=-padding; i <= padding; i++)
            for(int j=-padding; j <= padding; j++)
            {
                float neighbor_pixel = buffer[(ty0 + i) * shared_size +(tx0+j)];                
                kernel_sum += neighbor_pixel* D_EDGE[(i+padding) * kernel_size + (j + padding)];
            }
        channel_output[idx] = central_pixel + alpha * (central_pixel - kernel_sum );
    }

}


/* This kernel applies Gaussian Blur to the image*/
__global__ void Gaussian_Blur_kernel(float* channel_input, float* channel_output, int shared_size, int padding, int kernel_size)
{
    int x=blockIdx.x * blockDim.x + threadIdx.x, y=blockIdx.y * blockDim.y + threadIdx.y;
    int idx= (y)* D_width  + (x);  
    //int shared_size = block_dim + 2*padding;
    
    int ty = threadIdx.y, tx = threadIdx.x;
    int ty0 = ty+padding, tx0 = tx+padding;
    extern __shared__ float buffer[];
    for(int l = 0; (ty + l * block_dim) < shared_size; l++)
        for(int m = 0; (tx + m * block_dim)< shared_size; m++)
        {
            int x0 = reflect_padding((x - padding) + m * block_dim , D_width );
            int y0 = reflect_padding((y - padding) + l * block_dim , D_length );

            buffer[(ty + l * block_dim) * shared_size + (tx + m * block_dim)] = channel_input[y0 * D_width + x0];
        }

    
    __syncthreads();
    if(y<D_length && x<D_width)
    {
        float kernel_sum = 0.0f;
        for(int i=-padding; i <= padding; i++)
            for(int j=-padding; j <= padding; j++)
            {
                
                float neighbor_pixel = buffer[(ty0 + i) * shared_size +(tx0+j)];                
                kernel_sum += neighbor_pixel* D_GAUSSIAN[(i+padding) * kernel_size + (j + padding)];
            }
        channel_output[idx] = kernel_sum;    
    }
}


// Host code for Image Signal Processing Pipeline
float ISP(uint64_t Input_image, uint64_t buffer_1, uint64_t buffer_2, uint64_t buffer_3, uint64_t buffer_4, uint64_t buffer_5, uint64_t buffer_6, uint64_t buffer_int_1, uint64_t buffer_int_2, uint64_t buffer_int_3, const configuration& cfg )    
{
    
    cudaEvent_t start, stop;
    // cudaEvent_t lap1,lap2,lap3,lap4,lap5,lap6,lap7,lap8,lap9,lap10,lap11,lap12;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // cudaEventCreate(&lap1);
    // cudaEventCreate(&lap2);
    // cudaEventCreate(&lap3);
    // cudaEventCreate(&lap4);
    // cudaEventCreate(&lap5);
    // cudaEventCreate(&lap6);
    // cudaEventCreate(&lap7);
    // cudaEventCreate(&lap8);
    // cudaEventCreate(&lap9);
    // cudaEventCreate(&lap10);
    // cudaEventCreate(&lap11);
    // cudaEventCreate(&lap12);
    float ms;
    

    cudaEventRecord(start);

    ////////////////////////////////////////////////////
    //
    //
    // ### loading input image ###
    //
    //
    ////////////////////////////////////////////////////
    float  *D_Image_1 = reinterpret_cast<float *> (Input_image);
    float  *D_image_2 = reinterpret_cast<float *> (buffer_6);
    int array_size = cfg.width * cfg.length;

    load_data(cfg);

    ////////////////////////////////////////////////////
    //
    //
    // ### creating Device variables for loading data to GPU. D_Image_1 and D_Image_2 are image input output pairs which change job after each kernel. ###
    //
    //
    ////////////////////////////////////////////////////
    cudaMemcpyToSymbol(D_width, &cfg.width,  1 * sizeof(int));
    cudaMemcpyToSymbol(D_length, &cfg.length, 1 * sizeof(int));

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

    ////////////////////////////////////////////////////
    //
    //
    // ### pipeline starts. ###
    // Executing Dead Pixel Correction Kernel
    //
    ////////////////////////////////////////////////////
    // cudaEventRecord(lap1);
    // cudaEventSynchronize(lap1);
    // cudaEventElapsedTime(&ms, start, lap1);
    // printf("Pipeline GPU time initialization: %.3f ms\n", ms);
    
    if(cfg.DPC)
    {
        DP_kernel<<<dim3(blockx,blocky),dim3(block_dim,block_dim)>>>(D_Image_1,D_image_2, cfg.DPC_threshold);          //calling __global__ function (CUDA kernel)
        //cudaDeviceSynchronize(); //wait until all kernels stop executing.
    }
    else
    {
        cudaMemcpy( D_image_2, D_Image_1, array_size * sizeof(float), cudaMemcpyDeviceToDevice);
    }



    // cudaEventRecord(lap2);
    // cudaEventSynchronize(lap2);
    // cudaEventElapsedTime(&ms, lap1, lap2);
    // printf("Pipeline GPU time Defective Pixel Correction: %.3f ms\n", ms);

    ////////////////////////////////////////////////////
    //
    //
    // Executing Black Level Correction Kernel and Lens shading correction
    //
    //
    ////////////////////////////////////////////////////
    if(cfg.BLC || cfg.LSC)
    {

        BLC_LSC_kernel<<<dim3(blockx,blocky),dim3(block_dim,block_dim)>>>(D_image_2, cfg.BLC, cfg.LSC, cfg.LSC_Max_radius );
        //cudaDeviceSynchronize();
    }



    // cudaEventRecord(lap3);
    // cudaEventSynchronize(lap3);
    // cudaEventElapsedTime(&ms, lap2, lap3);
    // printf("Pipeline GPU time Lens shading correction: %.3f ms\n", ms);
    ////////////////////////////////////////////////////
    //
    //
    // Executing Automatic white balance gain
    //
    //
    ////////////////////////////////////////////////////
    double* D_AWBG;
    if(cfg.AWB)
    {
        float GAIN_RED;
        float GAIN_GREEN;
        float GAIN_BLUE;
        if(!cfg.AWB_Value_Given)
        {
            
            double H_AWBG[3];

            cudaMalloc( &D_AWBG, 3 * sizeof(double));
            cudaMemset( D_AWBG, 0, 3 * sizeof(double));

            AWBG_kernel<<<dim3(blockx,blocky),dim3(block_dim,block_dim)>>>( D_image_2, D_AWBG,cfg.orientation);
            cudaDeviceSynchronize();

            cudaMemcpy(H_AWBG, D_AWBG, 3 * sizeof(double), cudaMemcpyDeviceToHost);

            GAIN_RED  = (float)((double)(H_AWBG[0])/(double)(2*H_AWBG[2]));
            GAIN_GREEN = 1.0f;
            GAIN_BLUE = (float)((double)(H_AWBG[0])/(double)(2*H_AWBG[1]));
            

        }
        else
        {
            GAIN_RED = cfg.AWB_gain[0];
            GAIN_GREEN = cfg.AWB_gain[1];
            GAIN_BLUE = cfg.AWB_gain[2];
        }

        switch (cfg.orientation)
        {
        case 0:
            /* code */
            if(cfg.Exposure)
        {
            float compensation = pow(2.0f, cfg.Exposure_value);
            AWBG_Apply_kernel<<<dim3(blockx,blocky),dim3(block_dim,block_dim)>>>( D_image_2, GAIN_RED, GAIN_GREEN, GAIN_BLUE,compensation, 0);
            //cudaDeviceSynchronize();
        }
        else 
        {
            AWBG_Apply_kernel<<<dim3(blockx,blocky),dim3(block_dim,block_dim)>>>( D_image_2, GAIN_RED, GAIN_GREEN, GAIN_BLUE, 0);
        }

            break;

        case 1:
            /* code */
            if(cfg.Exposure)
            {
                float compensation = pow(2.0f, cfg.Exposure_value);
                AWBG_Apply_kernel<<<dim3(blockx,blocky),dim3(block_dim,block_dim)>>>( D_image_2, GAIN_RED, GAIN_GREEN, GAIN_BLUE,compensation, 1);
                //cudaDeviceSynchronize();
            }
            else 
            {
                AWBG_Apply_kernel<<<dim3(blockx,blocky),dim3(block_dim,block_dim)>>>( D_image_2, GAIN_RED, GAIN_GREEN, GAIN_BLUE, 1);
            }
            break;
        case 2:
            /* code */
            if(cfg.Exposure)
            {
                float compensation = pow(2.0f, cfg.Exposure_value);
                AWBG_Apply_kernel<<<dim3(blockx,blocky),dim3(block_dim,block_dim)>>>( D_image_2, GAIN_RED, GAIN_GREEN, GAIN_BLUE,compensation, 2);
                //cudaDeviceSynchronize();
            }
            else 
            {
                AWBG_Apply_kernel<<<dim3(blockx,blocky),dim3(block_dim,block_dim)>>>( D_image_2, GAIN_RED, GAIN_GREEN, GAIN_BLUE, 2);
            }
            break;
        case 3:
            /* code */
            if(cfg.Exposure)
            {
                float compensation = pow(2.0f, cfg.Exposure_value);
                AWBG_Apply_kernel<<<dim3(blockx,blocky),dim3(block_dim,block_dim)>>>( D_image_2, GAIN_RED, GAIN_GREEN, GAIN_BLUE,compensation, 3);
                //cudaDeviceSynchronize();
            }
            else 
            {
                AWBG_Apply_kernel<<<dim3(blockx,blocky),dim3(block_dim,block_dim)>>>( D_image_2, GAIN_RED, GAIN_GREEN, GAIN_BLUE, 3);
            }
            break;
        
        default:
            break;
        }



        
    }

    // cudaEventRecord(lap4);
    // cudaEventSynchronize(lap4);
    // cudaEventElapsedTime(&ms, lap3, lap4);
    // printf("Pipeline GPU time White balance: %.3f ms\n", ms);

    ////////////////////////////////////////////////////
    //
    //
    //Executing De-Bayer kernels
    //This is not an optional kernel as debayering produces 3 color channel output. This stage is required for the following stages.
    //                      Channel_0 :: Red   Channel_1 :: Green    Channel_2 :: Red
    //
    ////////////////////////////////////////////////////
    float* CHANNEL_0 = reinterpret_cast<float *> (buffer_1);
    float* CHANNEL_1 = reinterpret_cast<float *> (buffer_2);
    float* CHANNEL_2 = reinterpret_cast<float *> (buffer_3);

    int* RED = reinterpret_cast<int *> (buffer_int_1) ;
    int* GREEN = reinterpret_cast<int *> (buffer_int_2);
    int* BLUE =reinterpret_cast<int *> (buffer_int_3);

    int padding = 2;
    int shared_size = block_dim + 2 * padding;
    int shared_vol = shared_size * shared_size ;
    int shared_memory_vol = shared_vol * sizeof(float);


    switch (cfg.orientation)
    {
        case 0:
            /* code */
            DEBAYER_kernel_1<<<dim3(blockx,blocky),dim3(block_dim ,block_dim), shared_memory_vol>>>(D_image_2, CHANNEL_1, shared_size, padding, 0); // the no of threads are increased to 400 per block for acting as halo and padding for the block level operations
            //cudaDeviceSynchronize();

            shared_memory_vol = shared_memory_vol * 2;

            DEBAYER_kernel_2<<<dim3(blockx,blocky),dim3(block_dim ,block_dim), shared_memory_vol>>>(D_image_2, CHANNEL_1, CHANNEL_0,  CHANNEL_2, shared_size, shared_vol,  padding, 0);
        //cudaDeviceSynchronize();
            break;
        case 1:
            DEBAYER_kernel_1<<<dim3(blockx,blocky),dim3(block_dim ,block_dim), shared_memory_vol>>>(D_image_2,CHANNEL_1, shared_size, padding, 1); // the no of threads are increased to 400 per block for acting as halo and padding for the block level operations
            //cudaDeviceSynchronize();

            shared_memory_vol = shared_memory_vol * 2;

            DEBAYER_kernel_2<<<dim3(blockx,blocky),dim3(block_dim ,block_dim), shared_memory_vol>>>(D_image_2, CHANNEL_1, CHANNEL_0,  CHANNEL_2, shared_size, shared_vol,  padding, 1);
        //cudaDeviceSynchronize();
            break;
        case 2:
            DEBAYER_kernel_1<<<dim3(blockx,blocky),dim3(block_dim ,block_dim), shared_memory_vol>>>(D_image_2, CHANNEL_1, shared_size, padding, 2); // the no of threads are increased to 400 per block for acting as halo and padding for the block level operations
            //cudaDeviceSynchronize();

            shared_memory_vol = shared_memory_vol * 2;

            DEBAYER_kernel_2<<<dim3(blockx,blocky),dim3(block_dim ,block_dim), shared_memory_vol>>>(D_image_2, CHANNEL_1, CHANNEL_0,  CHANNEL_2, shared_size, shared_vol,  padding, 2);
            //cudaDeviceSynchronize();
            /* code */
            break;
        case 3:
            DEBAYER_kernel_1<<<dim3(blockx,blocky),dim3(block_dim ,block_dim), shared_memory_vol>>>(D_image_2,  CHANNEL_1, shared_size, padding, 3); // the no of threads are increased to 400 per block for acting as halo and padding for the block level operations
            //cudaDeviceSynchronize();

            shared_memory_vol = shared_memory_vol * 2;

            DEBAYER_kernel_2<<<dim3(blockx,blocky),dim3(block_dim ,block_dim), shared_memory_vol>>>(D_image_2, CHANNEL_1, CHANNEL_0,  CHANNEL_2, shared_size, shared_vol,  padding, 3);
            //cudaDeviceSynchronize();
            /* code */
            break;
        
        default:
            break;
    }


    // cudaEventRecord(lap5);
    // cudaEventSynchronize(lap5);
    // cudaEventElapsedTime(&ms, lap4, lap5);
    // printf("Pipeline GPU time debayering: %.3f ms\n", ms);


    ////////////////////////////////////////////////////
    //
    //
    //freeing D_image_2 as the source image is no longer required so making it a temporary buffer for future calculations.
    //
    //
    ////////////////////////////////////////////////////
    float* channel_temp_0 = reinterpret_cast<float *> (buffer_4);
    float* channel_temp_1 = reinterpret_cast<float *> (buffer_5);
    float* channel_temp_2 = D_image_2;

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
        Transform_Kernel<<<dim3(blockx,blocky),dim3(block_dim,block_dim)>>>(CHANNEL_0, CHANNEL_1, CHANNEL_2);
        //cudaDeviceSynchronize();
    }

    // cudaEventRecord(lap6);
    // cudaEventSynchronize(lap6);
    // cudaEventElapsedTime(&ms, lap5, lap6);
    // printf("Pipeline GPU time CCM: %.3f ms\n", ms);


    ////////////////////////////////////////////////////
    //
    //
    // Color Space Conversion to YCbCr
    //  Channel_0 :: Y   Channel_1 :: Cb    Channel_2 :: Cr
    //
    ////////////////////////////////////////////////////
    if(cfg.Color_Space_Conversion)
    {
        Transform_Kernel_RGB_YCbCr<<<dim3(blockx,blocky),dim3(block_dim,block_dim)>>>(CHANNEL_0, CHANNEL_1, CHANNEL_2);
        //cudaDeviceSynchronize();
    }

    // cudaEventRecord(lap7);
    // cudaEventSynchronize(lap7);
    // cudaEventElapsedTime(&ms, lap6, lap7);
    // printf("Pipeline GPU time CSC 1: %.3f ms\n", ms);


    ////////////////////////////////////////////////////
    //
    //
    // edge aware low pass filtering - Bilateral kernel on luminance channel
    //  Channel_0 :: Y   Channel_1 :: Cb    Channel_2 :: Cr
    //
    ////////////////////////////////////////////////////


    if(cfg.Color_Space_Conversion && cfg.Bilateral_Filter)
    {
        if(cfg.Joint_bilateral_kernel)
        {
            
            int padding = cfg.Bilateral_kernel_size /2;
            int shared_size = block_dim + 2 * padding;
            int shared_vol = shared_size * shared_size ;
            int shared_memory_vol = 3 * shared_vol * sizeof(float);
            float spatial_pdt =  1.0f / (2.0f *cfg.Bilateral_spatial_STD*cfg.Bilateral_spatial_STD);
            float range_pdt = 1.0f / (2.0f *cfg.Bilateral_Range_STD * cfg.Bilateral_Range_STD);
            Bilateral_filter_kernel<<<dim3(blockx,blocky),dim3(block_dim,block_dim),  shared_memory_vol>>>( CHANNEL_0, CHANNEL_1, CHANNEL_2, channel_temp_0, channel_temp_1, channel_temp_2, shared_size, shared_vol, padding, spatial_pdt , range_pdt);
            //cudaDeviceSynchronize();

            float* buf = channel_temp_0;
            channel_temp_0 = CHANNEL_0;
            CHANNEL_0 = buf;

            buf = channel_temp_1;
            channel_temp_1 = CHANNEL_1;
            CHANNEL_1 = buf;

            buf = channel_temp_2;
            channel_temp_2 = CHANNEL_2;
            CHANNEL_2 = buf;

        }
        else
        {

            int padding = cfg.Bilateral_kernel_size /2;
            int shared_size = block_dim + 2 * padding;
            size_t shared_memory_vol = shared_size * shared_size * sizeof(float);
            float spatial_pdt =  1.0f / (2.0f *cfg.Bilateral_spatial_STD*cfg.Bilateral_spatial_STD);
            float range_pdt = 1.0f / (2.0f *cfg.Bilateral_Range_STD * cfg.Bilateral_Range_STD);
            Bilateral_filter_kernel<<<dim3(blockx,blocky),dim3(block_dim,block_dim), shared_memory_vol>>>( CHANNEL_0, channel_temp_0, shared_size, padding, spatial_pdt , range_pdt);
            //cudaDeviceSynchronize();

            float* buf = channel_temp_0;
            channel_temp_0 = CHANNEL_0;
            CHANNEL_0 = buf;
        }

    }
    
    // cudaEventRecord(lap8);
    // cudaEventSynchronize(lap8);
    // cudaEventElapsedTime(&ms, lap7, lap8);
    // printf("Pipeline GPU time Bilateral: %.3f ms\n", ms);

    ////////////////////////////////////////////////////
    //
    //
    //  Gaussian Blur
    //  Channel_0 :: Y   Channel_1 :: Cb    Channel_2 :: Cr
    //
    ////////////////////////////////////////////////////

    if(cfg.Color_Space_Conversion && cfg.Gaussian_blur )
    {


        int padding = cfg.Gaussian_blur_kernel_size /2;
        int shared_size = block_dim + 2 * padding;
        const int kernel_vol = cfg.Gaussian_blur_kernel_size * cfg.Gaussian_blur_kernel_size;
        size_t shared_memory_vol = shared_size * shared_size * sizeof(float);

        float gaussian[256], sum=0.0f;

        for(int i=-padding ; i <= padding ; i++)
            for(int j=-padding ; j <= padding ; j++)
            {
                gaussian[(i+padding) * cfg.Gaussian_blur_kernel_size + (j+padding)] = std:: exp(-(i*i + j*j)/(2.0f * cfg.Gaussian_STD * cfg.Gaussian_STD) );
                sum+=gaussian[(i+padding) * cfg.Gaussian_blur_kernel_size + (j+padding)];
            }
        for(int i=0 ; i < kernel_vol ; i++) gaussian[i]/=sum;

        cudaMemcpyToSymbol(D_GAUSSIAN, gaussian, kernel_vol * sizeof(float));

        Gaussian_Blur_kernel<<<dim3(blockx,blocky),dim3(block_dim,block_dim), shared_memory_vol>>>( CHANNEL_0, channel_temp_0, shared_size, padding, cfg.Gaussian_blur_kernel_size);
        //cudaDeviceSynchronize();

        float* buf = channel_temp_0;
        channel_temp_0 = CHANNEL_0;
        CHANNEL_0 = buf;

    }
    // cudaEventRecord(lap9);
    // cudaEventSynchronize(lap9);
    // cudaEventElapsedTime(&ms, lap8, lap9);
    // printf("Pipeline GPU time Gaussian: %.3f ms\n", ms);

    

    ////////////////////////////////////////////////////
    //
    //
    // edge enhancement using High Boost filter( unsharp mask )
    //  Channel_0 :: Y   Channel_1 :: Cb    Channel_2 :: Cr
    //
    ////////////////////////////////////////////////////

    if(cfg.Color_Space_Conversion && cfg.Edge_enhancement  )
    {


        int padding = cfg.Edge_enhancement_kernel_size /2;
        int shared_size = block_dim + 2 * padding;
        const int kernel_vol = cfg.Edge_enhancement_kernel_size * cfg.Edge_enhancement_kernel_size;
        size_t shared_memory_vol = shared_size * shared_size * sizeof(float);

        float gaussian[256], sum=0.0f;
        for(int i=-padding ; i <= padding ; i++)
            for(int j=-padding ; j <= padding ; j++)
            {
                gaussian[(i+padding) * cfg.Edge_enhancement_kernel_size + (j+padding)] = std:: exp(-(i*i + j*j)/(2.0f * cfg.Edge_enhancement_STD * cfg.Edge_enhancement_STD));
                sum+=gaussian[(i+padding) * cfg.Edge_enhancement_kernel_size + (j+padding)];
            }
        for(int i=0 ; i < kernel_vol ; i++) gaussian[i]/=sum;
        
        cudaMemcpyToSymbol(D_EDGE, gaussian, kernel_vol * sizeof(float));

        Edge_enhancement_kernel<<<dim3(blockx,blocky),dim3(block_dim,block_dim), shared_memory_vol>>>( CHANNEL_0, channel_temp_0, shared_size, padding, cfg.Edge_enhancement_kernel_size,  cfg.Edge_enhancement_A_Value);
        //cudaDeviceSynchronize();

        float* buf = channel_temp_0;
        channel_temp_0 = CHANNEL_0;
        CHANNEL_0 = buf;

    }

    // cudaEventRecord(lap10);
    // cudaEventSynchronize(lap10);
    // cudaEventElapsedTime(&ms, lap9, lap10);
    // printf("Pipeline GPU time Edge enhancement: %.3f ms\n", ms);

    ////////////////////////////////////////////////////
    //
    //
    //  Brightness , Hue, contrast, tint, vibrance and Saturation Adjustment
    //  Channel_0 :: Y   Channel_1 :: Cb    Channel_2 :: Cr
    //
    ////////////////////////////////////////////////////
    if(cfg.Color_Space_Conversion && (cfg.Brightness|| cfg.Saturation || cfg.Hue || cfg.Contrast || cfg.Tint || cfg.Vibrance))
    {
        float Hue_mat[4];
        for(int i=0;i < 4; i++)
        {
            Hue_mat[i] = (i == 0 || i == 3) ? cos(cfg.Hue_value) : ((i==1) ? -sin(cfg.Hue_value): sin(cfg.Hue_value));
        }
        cudaMemcpyToSymbol(D_hue, Hue_mat, 4 * sizeof(float));

        float Hvalue =  0.5f * cfg.white_level;

        Color_control_kernel<<<dim3(blockx,blocky),dim3(block_dim,block_dim)>>>(CHANNEL_0,CHANNEL_1,CHANNEL_2, cfg.Brightness_value, cfg.Saturation_value, cfg.Contrast_value, cfg.Tint_value, cfg.Vibrance_value, cfg.Brightness, cfg.Saturation, cfg.Hue, cfg.Contrast, cfg.Tint, cfg.Vibrance, Hvalue);
        //cudaDeviceSynchronize();
        

    }

    // cudaEventRecord(lap11);
    // cudaEventSynchronize(lap11);
    // cudaEventElapsedTime(&ms, lap10, lap11);
    // printf("Pipeline GPU time Tone control: %.3f ms\n", ms);

    ////////////////////////////////////////////////////
    //
    //
    // Color Space Conversion to RGB
    //  Channel_0 :: R   Channel_1 :: G    Channel_2 :: B
    //
    ////////////////////////////////////////////////////
    if(cfg.Color_Space_Conversion)
    {
        Transform_Kernel_YCbCr_RGB<<<dim3(blockx,blocky),dim3(block_dim,block_dim)>>>(CHANNEL_0, CHANNEL_1, CHANNEL_2);
        //cudaDeviceSynchronize();
    }


    // cudaEventRecord(lap12);
    // cudaEventSynchronize(lap12);
    // cudaEventElapsedTime(&ms, lap11, lap12);
    // printf("Pipeline GPU time csc 2: %.3f ms\n", ms);

    ////////////////////////////////////////////////////
    //
    //
    // Executing Gamma correction using Lookup Table
    //
    //
    ////////////////////////////////////////////////////
    if(cfg.GAMMA)
    {
        float x;
      

        std::vector<int> LUT(256);

        for(int i=0;i<256;i++)
        {
            x = (float)(i)/(255.0);
            if(x<0.0031308f)
            {
                x = 12.92f * x;
            }
            else{
                x = 1.055f * powf(x , (1.0f/cfg.GAMMA_VALUE)) - 0.055f;
            }
            
            LUT[i] = (int)roundf(fmaxf(0.0f,fminf(255.0,x * 255.0f)));
        }

        

        cudaMemcpyToSymbol(D_LUT, LUT.data() , (256) *  sizeof(int));
        
        LUT_kernel_Gamma<<<dim3(blockx,blocky),dim3(block_dim,block_dim)>>>( CHANNEL_0, CHANNEL_1, CHANNEL_2, RED, GREEN, BLUE, 255.0f / cfg.white_level);


        cudaDeviceSynchronize();
    }

    
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&ms, lap12, stop);
    // printf("Pipeline GPU time Gamma: %.3f ms\n", ms);


    if(!cfg.GAMMA)
    {    
        Norm_kernel<<<dim3(blockx,blocky),dim3(block_dim,block_dim)>>>( CHANNEL_0, CHANNEL_1, CHANNEL_2, RED, GREEN, BLUE, 255.0f / cfg.white_level);
        cudaDeviceSynchronize();
    }

    cudaFree(D_AWBG);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("Pipeline GPU time Total: %.3f ms\n", ms);

    return ms;
    
}

PYBIND11_MODULE(ISP, m) {
    
    py::class_<configuration>(m, "Configuration",
        R"pbdoc(
    ISP Configuration Structure

    Attributes
    ----------
    D_width : int
        Image D_width.

    D_length : int
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
        .def_readwrite("AWB_Value_Given", &configuration::AWB_Value_Given)
        .def_readwrite("Color_Space_Conversion", &configuration::Color_Space_Conversion)
        .def_readwrite("Brightness", &configuration::Brightness)
        .def_readwrite("Brightness_value", &configuration::Brightness_value)
        .def_readwrite("Saturation", &configuration::Saturation)
        .def_readwrite("Saturation_value", &configuration::Saturation_value)
        .def_readwrite("Hue", &configuration::Hue)
        .def_readwrite("Hue_value", &configuration::Hue_value)
        .def_readwrite("Contrast", &configuration::Contrast)
        .def_readwrite("Contrast_value", &configuration::Contrast_value)
        .def_readwrite("Tint", &configuration::Tint)
        .def_readwrite("Tint_value", &configuration::Tint_value)
        .def_readwrite("Vibrance", &configuration::Vibrance)
        .def_readwrite("Vibrance_value", &configuration::Vibrance_value)
        .def_readwrite("Bilateral_Filter", &configuration::Bilateral_Filter)
        .def_readwrite("Bilateral_kernel_size", &configuration::Bilateral_kernel_size)
        .def_readwrite("Bilateral_spatial_STD", &configuration::Bilateral_spatial_STD)
        .def_readwrite("Bilateral_Range_STD", &configuration::Bilateral_Range_STD)
        .def_readwrite("Edge_enhancement", &configuration::Edge_enhancement)
        .def_readwrite("Edge_enhancement_A_Value", &configuration::Edge_enhancement_A_Value)
        .def_readwrite("Edge_enhancement_kernel_size", &configuration::Edge_enhancement_kernel_size)
        .def_readwrite("Edge_enhancement_STD", &configuration::Edge_enhancement_STD)
        .def_readwrite("Gaussian_blur", &configuration::Gaussian_blur)
        .def_readwrite("Gaussian_STD", &configuration::Gaussian_STD)
        .def_readwrite("Exposure", &configuration::Exposure)
        .def_readwrite("Exposure_value", &configuration::Exposure_value)
        .def_readwrite("Gaussian_blur_kernel_size", &configuration::Gaussian_blur_kernel_size)
        .def_readwrite("Joint_bilateral_kernel", &configuration::Joint_bilateral_kernel);

    m.def("ISP", &ISP, "Image Signal Processing Pipeline");
}


