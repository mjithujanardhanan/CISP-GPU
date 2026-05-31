
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

__constant__ float d_CCM[9];

struct configuration
{
    int length;                                                                                                                                                                             // length of image
    int width;                                                                                                                                                                              // width of image
    int DPC_threshold;                                                                                                                                                                      // Threshold for Dead pixel correction
    int Bayer_orientation;                                                                                                                                                                  // BGGR - 0,  GBRG -1, GRBG -2, RGGB -3 

    std::array<int, 4> Black_Level;

    float LSC_gain_00;                                                                                                                                                                      //Lens shading correction gains for each channel
    float LSC_gain_01;
    float LSC_gain_02;
    float LSC_gain_03;

    float Gamma_value = 2.4;

    std::vector<float> CCM;
};



/* this program is an execution of Defective Pixel Consealment on digital bayer domain images*/
__global__ void DP_kernel(int* Image , int* image_out, int width, int Length, int threshold)                                                                                                // Image - the image on which the operation is to be performed. Image out- the output image. Threshold- the threshold for dpc correction
{

    // directional gradient calculation
    long j=blockIdx.x * blockDim.x + threadIdx.x, i=blockIdx.y * blockDim.y + threadIdx.y;                                                                                                  // i,j are 2d image coordinates calculated from the thread id
    long idx= (i)* width  + (j);                                                                                                                                                            // idx is the id of (i,j) element in the flattened image
    if(i<Length && j<width)                                                                                                                                                                 // out of bounds check
    {

        int up = (i-2)<0?i+2:i-2;                                                                                                                                                           //reflective padding implementation.(important in dpc calculation)
        int down = (i+2)>=Length?i-2:i+2;
        int left = (j-2)<0?j+2:j-2;
        int right=(j+2)>=width?j-2:j+2;

        int p1 = up * width + j;                                                                                                                                                            //assigning ids to neighbor elements ( future updation to shared memory will remove this part)
        int p2 = up * width + left;
        int p3 = i * width + left;
        int p4 = down * width + left;
        int p5 = down * width + j;
        int p6 = down * width + right;
        int p7 = i * width + right;
        int p8 = up * width + right;


        int d1,d2,d3,d4;                                                                                                                                                                    //gradient calculation
        d1 = abs(Image[p5]-Image[p1]);
        d2 = abs(Image[p6]-Image[p2]);
        d3 = abs(Image[p7]-Image[p3]);
        d4 = abs(Image[p8]-Image[p4]);

        int min=d1, neighbor_avg = (Image[p5]+Image[p1])>>1;                                                                                                                               //finding min neighbor_average. most similarity will be along the direction of least gradient.
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

        if(abs(Image[idx] -neighbor_avg) >threshold)                                                                                                                                        //DPC in effect. check against threshold to classify and replace the Dead Pixel.
        {
            image_out[idx] = neighbor_avg ;
        }

        else 
        {
            image_out[idx] = Image[idx] ;
        }

    }



}


/* this program is an execution of lens shading correction on digital bayer domain images*/
__global__ void LSC_kernel(int* Image , int width, int Length, float gain_00, float gain_01, float gain_10, float gain_11, float Max_radius)                                                // gain is for every color in bayer format image assed in the input configuration. 
{

    // Lens Shading Correction calculation
    long j=blockIdx.x * blockDim.x + threadIdx.x, i=blockIdx.y * blockDim.y + threadIdx.y;
    if(i<Length && j<width)                                                                                                                                                                 //boundary check
    {
        long idx= (i)* width  + (j);
        float a[2][2]={{gain_00,gain_01},{gain_10,gain_11}};
        float dx = float(width)/2.0f -j;                                                                                                                                                    //dx distance from centre to x (here x is j)
        float dy = float(Length)/2.0f -i;                                                                                                                                                   //dy distance from centre to y (here y is i)
        float r = sqrtf(dx*dx + dy*dy);                                                                                                                                                     //radius r calculation
        
        Image[idx] = (int)(Image[idx]*( 1.0f + r*a[i%2][j%2]/ Max_radius));                                                                                                                 //lens shading correction modelled as a linear function (original lens shading is modelled as a cos^4 function which will be implemented in a future version. this is adopted only for development purpose)

    }



}


/* this program is an execution of Black Level coreection on digital bayer domain images*/
__global__ void BLC_kernel(int* Image , int Even_0, int Even_1, int Odd_0, int Odd_1, int width, int length)
{

    int j=blockIdx.x * blockDim.x + threadIdx.x, i=blockIdx.y * blockDim.y + threadIdx.y;
    int idx= (i)* width  + (j);
    if(i<length && j<width)
    {

       int offset[2][2]={{Even_0,Even_1},{Odd_0,Odd_1}};


       int val = Image[idx] - offset[i%2][j%2];

       Image[idx] = (val>0)?val:0;

    }



}


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

// kernel for debayering the input image 
__global__ void DEBAYER_kernel_1(int* Image , int* output, int orientation,int width, int length)
{
    int j=blockIdx.x * block_dim + threadIdx.x, i=blockIdx.y * block_dim + threadIdx.y;
    int idx= abs(i-=2)* width  + abs(j-=2);

    __shared__ float buffer[20][21];

    int tj=threadIdx.x, ti=threadIdx.y;

    if(i<length && j<width)
    {
        buffer[ti][tj] = Image[idx];
    }
    else if(i<length+2 && j<width+2)
    {
        
        if(length+2-i == 1) i=length-2;
        else i=length-3;

        if(width+2-j == 1) j=width-2;
        else j=width-3;

        idx= abs(i)* width  + abs(j);
        buffer[ti][tj] = Image[idx];

    }

    __syncthreads(); 

    if(ti>1 && ti<18 && tj>1 && tj<18)
    {
        int j=blockIdx.x * block_dim + tj-2, i=blockIdx.y * block_dim + ti-2;
        if(i<length && j<width)
        {
            int idx= i* width  + j;
            if(orientation == 0 || orientation == 3)
            {
                if((i+j)&1)
                {
                    output[idx]=Image[idx];
                }
                else
                {
                    float dv = fabsf(buffer[ti][tj-1] - buffer[ti][tj+1]) + fabsf(2* buffer[ti][tj] -(buffer[ti][tj-2] + buffer[ti][tj+2]));
                    float dh = fabsf(buffer[ti-1][tj] - buffer[ti+1][tj]) + fabsf(2* buffer[ti][tj] -(buffer[ti-2][tj] + buffer[ti+2][tj]));

                    if (dh>dv)
                    {
                        output[idx] = (buffer[ti][tj-1] + buffer[ti][tj+1])*0.5 + (2* buffer[ti][tj] -(buffer[ti][tj-2] + buffer[ti][tj+2]))*0.25;
                    }
                    
                    else if (dh<dv)
                    {
                        output[idx] = (buffer[ti-1][tj] + buffer[ti+1][tj])*0.5 + (2* buffer[ti][tj] -(buffer[ti-2][tj] + buffer[ti+2][tj]))*0.25;
                    }
                    else
                    {
                        output[idx] = (buffer[ti][tj-1] + buffer[ti][tj+1] + buffer[ti-1][tj] + buffer[ti+1][tj])*0.25 + (2* buffer[ti][tj] -(buffer[ti][tj-2] + buffer[ti][tj+2]) +2* buffer[ti][tj] - (buffer[ti-2][tj] + buffer[ti+2][tj]))*0.125;
                    }
                    
                }
            }

            if(orientation == 1 || orientation == 2)
            {
                if(!((i+j)&1))
                {
                    output[idx]=buffer[ti][tj];
                }
                else
                {
                    float dv = fabsf(buffer[ti][tj-1] - buffer[ti][tj+1]) + fabsf(2* buffer[ti][tj] -(buffer[ti][tj-2] + buffer[ti][tj+2]));
                    float dh = fabsf(buffer[ti-1][tj] - buffer[ti+1][tj]) + fabsf(2* buffer[ti][tj] -(buffer[ti-2][tj] + buffer[ti+2][tj]));

                    if (dh>dv)
                    {
                        output[idx] = (buffer[ti][tj-1] + buffer[ti][tj+1])*0.5+ (2* buffer[ti][tj] -(buffer[ti][tj-2] + buffer[ti][tj+2]))*0.25;
                    }
                    
                    else if (dh<dv)
                    {
                        output[idx] = (buffer[ti-1][tj] + buffer[ti+1][tj])*0.5 + (2* buffer[ti][tj] -(buffer[ti-2][tj] + buffer[ti+2][tj]))*0.25;
                    }
                    else
                    {
                        output[idx] = (buffer[ti][tj-1] + buffer[ti][tj+1] + buffer[ti-1][tj] + buffer[ti+1][tj])*0.25 + (2* buffer[ti][tj] -(buffer[ti][tj-2] + buffer[ti][tj+2]) +2* buffer[ti][tj] -(buffer[ti-2][tj] + buffer[ti+2][tj]))*0.125;
                    }
                    
                }
            }
        }
    }
}

__global__ void DEBAYER_kernel_2(int* Image , int* green, int* red, int* blue, int orientation,int width, int length)
{
    int j=blockIdx.x * block_dim + threadIdx.x, i=blockIdx.y * block_dim + threadIdx.y;
    i -= 2;
    j -= 2;

    int idx = abs(i) * width + abs(j);

    __shared__ float buffer1[20][21], bufferg[20][21] ;

    int tj=threadIdx.x, ti=threadIdx.y;

    if(i<length && j<width)
    {
        buffer1[ti][tj] = Image[idx];
        bufferg[ti][tj] = green[idx];
    }
    else if(i<length+2 && j<width+2)
    {
        
        if(length+2-i == 1) i=length-2;
        else i=length-3;

        if(width+2-j == 1) j=width-2;
        else j=width-3;

        idx= abs(i)* width  + abs(j);
        buffer1[ti][tj] = Image[idx];
        bufferg[ti][tj] = green[idx];

    }

    __syncthreads(); 

    if(ti>1 && ti<18 && tj>1 && tj<18)
    {
        int j=blockIdx.x * block_dim + tj-2, i=blockIdx.y * block_dim + ti-2;
        if(i<length && j<width)
        {
            int idx= i* width  + j;
            if(orientation == 0 || orientation == 3)
            {
                if((i+j)&1)
                {
                    if(orientation ==0)
                    {
                        red[idx] = int (bufferg[ti][tj]+0.5f *(buffer1[ti-1][tj]-bufferg[ti-1][tj] + (buffer1[ti+1][tj]-bufferg[ti+1][tj])));       //vertical interpolation
                        blue[idx]=int(bufferg[ti][tj]+0.5f*(buffer1[ti][tj-1]-bufferg[ti][tj-1] + (buffer1[ti][tj+1]-bufferg[ti][tj+1])));        //horizontal interpolation
                    }
                    else
                    {
                        blue[idx] = int(bufferg[ti][tj]+0.5f*(buffer1[ti-1][tj]-bufferg[ti-1][tj] + (buffer1[ti+1][tj]-bufferg[ti+1][tj])));        //vertical interpolation
                        red[idx]  =int(bufferg[ti][tj]+0.5f*(buffer1[ti][tj-1]-bufferg[ti][tj-1] + (buffer1[ti][tj+1]-bufferg[ti][tj+1])));      //horizontal interpolation
                    }
                }
                else
                {
                    if(orientation ==0)
                    {
                        if(i&1 && j&1)
                        {
                            red[idx] = int(buffer1[ti][tj]);
                            blue[idx] = int(bufferg[ti][tj]+0.25f*(buffer1[ti-1][tj-1]-bufferg[ti-1][tj-1] + buffer1[ti+1][tj+1]-bufferg[ti+1][tj+1] +buffer1[ti+1][tj-1]-bufferg[ti+1][tj-1] + buffer1[ti-1][tj+1]-bufferg[ti-1][tj+1]));
                        }
                        else 
                        {
                            blue[idx] = int(buffer1[ti][tj]);
                            red[idx] = int(bufferg[ti][tj]+0.25f*(buffer1[ti-1][tj-1]-bufferg[ti-1][tj-1] + buffer1[ti+1][tj+1]-bufferg[ti+1][tj+1] +buffer1[ti+1][tj-1]-bufferg[ti+1][tj-1] + buffer1[ti-1][tj+1]-bufferg[ti-1][tj+1]));
                        }
            
                    }
                    else
                    {
                        if(i&1 && j&1)
                        {
                            blue[idx] = int(buffer1[ti][tj]);
                            red[idx] = int(bufferg[ti][tj]+0.25f*(buffer1[ti-1][tj-1]-bufferg[ti-1][tj-1] + buffer1[ti+1][tj+1]-bufferg[ti+1][tj+1] +buffer1[ti+1][tj-1]-bufferg[ti+1][tj-1] + buffer1[ti-1][tj+1]-bufferg[ti-1][tj+1]));
                        }
                        else 
                        {
                            red[idx] = int(buffer1[ti][tj]);
                            blue[idx] = int(bufferg[ti][tj]+0.25f*(buffer1[ti-1][tj-1]-bufferg[ti-1][tj-1] + buffer1[ti+1][tj+1]-bufferg[ti+1][tj+1] +buffer1[ti+1][tj-1]-bufferg[ti+1][tj-1] + buffer1[ti-1][tj+1]-bufferg[ti-1][tj+1]));
                        }


                    }
                    
                }
            }

            if(orientation == 1 || orientation == 2)
            {
                if(!((i+j)&1))
                {
                    if(orientation ==1)
                    {
                        red[idx] = int(bufferg[ti][tj]+0.5f * (buffer1[ti-1][tj]-bufferg[ti-1][tj] + (buffer1[ti+1][tj]-bufferg[ti+1][tj])));       //vertical interpolation
                        blue[idx]  =int(bufferg[ti][tj]+0.5f*(buffer1[ti][tj-1]-bufferg[ti][tj-1] + (buffer1[ti][tj+1]-bufferg[ti][tj+1])));        //horizontal interpolation
                    }
                    else
                    {
                        blue[idx] = int(bufferg[ti][tj]+0.5f*(buffer1[ti-1][tj]-bufferg[ti-1][tj] + (buffer1[ti+1][tj]-bufferg[ti+1][tj])));        //vertical interpolation
                        red[idx]  =int(bufferg[ti][tj]+0.5f*(buffer1[ti][tj-1]-bufferg[ti][tj-1] + (buffer1[ti][tj+1]-bufferg[ti][tj+1])));      //horizontal interpolation
                    }
                }
                else
                {
                    if(orientation ==1)
                    {
                        if(!(i&1) && j&1)
                        {
                            blue[idx] =int(buffer1[ti][tj]);
                            red[idx] = int(bufferg[ti][tj]+0.25f*(buffer1[ti-1][tj-1]-bufferg[ti-1][tj-1] + buffer1[ti+1][tj+1]-bufferg[ti+1][tj+1] +buffer1[ti+1][tj-1]-bufferg[ti+1][tj-1] + buffer1[ti-1][tj+1]-bufferg[ti-1][tj+1]));
                        }
                        else 
                        {
                            red[idx] = int(buffer1[ti][tj]);
                            blue[idx] = int(bufferg[ti][tj]+0.25f*(buffer1[ti-1][tj-1]-bufferg[ti-1][tj-1] + buffer1[ti+1][tj+1]-bufferg[ti+1][tj+1] +buffer1[ti+1][tj-1]-bufferg[ti+1][tj-1] + buffer1[ti-1][tj+1]-bufferg[ti-1][tj+1]));
                        }
            
                    }
                    else
                    {
                        if(!(i&1) && j&1)
                        {
                            red[idx] = int(buffer1[ti][tj]);
                            blue[idx] = int(bufferg[ti][tj]+0.25f*(buffer1[ti-1][tj-1]-bufferg[ti-1][tj-1] + buffer1[ti+1][tj+1]-bufferg[ti+1][tj+1] +buffer1[ti+1][tj-1]-bufferg[ti+1][tj-1] + buffer1[ti-1][tj+1]-bufferg[ti-1][tj+1]));
                        }
                        else 
                        {
                            blue[idx] = int(buffer1[ti][tj]);
                            red[idx] = int(bufferg[ti][tj]+0.25f*(buffer1[ti-1][tj-1]-bufferg[ti-1][tj-1] + buffer1[ti+1][tj+1]-bufferg[ti+1][tj+1] +buffer1[ti+1][tj-1]-bufferg[ti+1][tj-1] + buffer1[ti-1][tj+1]-bufferg[ti-1][tj+1]));
                        }


                    }
                    
                }
            }
        }
    }
}


//Kernel for gamma correction
__global__ void GAMMA_kernel(int* green, int* red, int* blue, unsigned char* LUT, int width, int length)
{
    int j=blockIdx.x * block_dim + threadIdx.x, i=blockIdx.y * block_dim + threadIdx.y;
    int idx= (i)* width  + (j);  
    if(i<length && j<width)
    {
        green[idx] = LUT[green[idx]];
        red[idx] = LUT[red[idx]];
        blue[idx] = LUT[blue[idx]];
    }
}

//Kernel for applying color correction matrix
__global__ void CCM_Kernel(int* green, int* red, int* blue,  int width, int length)
{
    int x=blockIdx.x * blockDim.x + threadIdx.x, y=blockIdx.y * blockDim.y + threadIdx.y;
    
    
    if(y<length && x<width)
    {
        int idx= (y)* width  + (x); 

        float Temp_R =    red[idx];
        float Temp_G =    green[idx];
        float Temp_B =    blue[idx];
        float R,G,B;

        R =      d_CCM[0] * Temp_R + d_CCM[1] * Temp_G + d_CCM[2] * Temp_B;
        G =    d_CCM[3] * Temp_R + d_CCM[4] * Temp_G + d_CCM[5] * Temp_B;
        B =     d_CCM[6] * Temp_R + d_CCM[7] * Temp_G + d_CCM[8] * Temp_B;

        red[idx] =          max(0, min(65535, (int)roundf(R)));
        green[idx] =        max(0, min(65535, (int)roundf(G)));
        blue[idx] =         max(0, min(65535, (int)roundf(B)));
    }
}

py::array_t<int> pipeline(py::array_t<int> Image, const configuration& cfg)    
{
   
    auto buffer = Image.request();

    if(buffer.ndim != 1)                                                                                                                                                                    //error check to see if flattened image is passed.
        throw std::runtime_error("image must be Flattened :: DPC module");
    if(buffer.size < (cfg.width * cfg.length))
        throw std::runtime_error("Wrong image size :: DPC module");
    
    int *Input = static_cast<int*>(buffer.ptr);
    long array_size = static_cast<int>(buffer.size);
    float max_radius = hypotf((cfg.width)/2.0f,float(cfg.length)/2.0f);

    int *D_Image_1, *D_image_2;                                                                                                                                                             //pointer declaration for gpu memory creation.
    unsigned long long* d_awbg;


    cudaMalloc( &D_Image_1, array_size * sizeof(int));                                                                                                                                      // creating memory pointers on gpu memory for image.
    cudaMalloc( &D_image_2, array_size * sizeof(int));
    cudaMalloc( &d_awbg, 3 * sizeof(unsigned long long));

    cudaMemset( d_awbg, 0, 3 * sizeof(unsigned long long));

    cudaMemcpy(D_Image_1 , Input, array_size * sizeof(int), cudaMemcpyHostToDevice);                                                                                                        //copying image data to gpu memory
    const int blockx= (width%16 == 0)?(width/16):(width/16 +1),blocky= (Length%16 == 0)?(Length/16):(Length/16 +1);

    // pipeline starts 

    DP_kernel<<<dim3(blockx,blocky),dim3(16,16)>>>(D_Image_1,D_image_2, cfg.width, cfg.length, cfg.DPC_threshold);                                                                          //calling __global__ function (CUDA kernel)
    cudaDeviceSynchronize(); //wait until all kernels stop executing.

    BLC_kernel<<<dim3(blockx,blocky),dim3(16,16)>>>(D_Image_2, cfg.Black_Level[0], cfg.Black_Level[1], cfg.Black_Level[2], cfg.Black_Level[3], cfg.width, cfg.length);
    cudaDeviceSynchronize();

    LSC_kernel<<<dim3(blockx,blocky),dim3(16,16)>>>(D_Image_2, cfg.width, cfg.length, cfg.LSC_gain_00,cfg.LSC_gain_01,cfg.LSC_gain_10,cfg.LSC_gain_11, max_radius);                         //calling __global__ function (CUDA kernel)
    cudaDeviceSynchronize();
    
    AWBG_kernel<<<dim3(blockx,blocky),dim3(16,16)>>>(D_Image_2 ,d_awbg, cfg.Bayer_orientation, cfg.width, cfg.length);
    cudaDeviceSynchronize();     
    unsigned long long h_awbg[3];
    cudaMemcpy(h_awbg, d_awbg, 3 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    float gain_r = float(h_awbg[1]) / float(h_awbg[0]);
    float gain_g = 1.0f;
    float gain_b = float(h_awbg[1]) / float(h_awbg[2]);
    AWBG_Apply_kernel<<<dim3(blockx,blocky),dim3(16,16)>>>(D_Image_2 , gain_r, gain_g, gain_b, cfg.Bayer_orientation, cfg.width, cfg.length);
    
    cudaFree(D_Image_1);

    int *D_Image_Gr, *D_image_Rd, *D_image_Bl;  // array for storing rgb after debayering

    cudaMalloc( &D_Image_Gr, array_size * sizeof(int));
    cudaMalloc( &D_image_Rd, array_size * sizeof(int));
    cudaMalloc( &D_image_Bl, array_size * sizeof(int));

    DEBAYER_kernel_1<<<dim3(blockx,blocky),dim3(20,20)>>>(D_Image_2, D_Image_Gr, cfg.Bayer_orientation, cfg.width, cfg.length);
    cudaDeviceSynchronize();
    DEBAYER_kernel_2<<<dim3(blockx,blocky),dim3(20,20)>>>(D_Image_2, D_Image_Gr ,D_image_Rd, D_image_Bl, d_awbg, cfg.Bayer_orientation, cfg.width, cfg.length);
    cudaDeviceSynchronize();

    cfg.Gamma_value = (1/cfg.Gamma_value);

    unsigned char *d_LUT;
    float x;
    cudaMalloc(&D_LUT, 65536 * sizeof(unsigned char));

    std::vector<uint16_t> LUT(65536);

    for(int i=0;i<65536;i++)
    {
        x = i/65535.0f
        x = powf(x, (cfg.Gamma_value));

        LUT[i] = (int)roundf(x*255);
    }

    cudaMemcpy(d_LUT, LUT , 65536 *  sizeof(unsigned char), cudaMemcpyHostToDevice);
    
    GAMMA_kernel<<<dim3(blockx,blocky), dim3(16,16)>>>(D_Image_Gr ,D_image_Rd, D_image_Bl, d_LUT, cfg.width, cfg.length );
    cudaDeviceSynchronize();

    CCM_Kernel<<<dim3(blockx,blocky),dim3(16,16)>>>(D_Image_Gr ,D_image_Rd, D_image_Bl)

    cudaMemcpy(Input,D_Image_2,array_size * sizeof(int),cudaMemcpyDeviceToHost);                                                                                                            // copy data back to ram memory.

    cudaFree(D_Image_2);                                                                                                                                                                    //destroy memory created in gpu.
    

    return Image;

}

PYBIND11_MODULE(pipeline, m) {
    // 1. Bind the configuration struct
    py::class_<configuration>(m, "Configuration")
        .def(py::init<>())
        .def_readwrite("length", &configuration::length)
        .def_readwrite("width", &configuration::width)
        .def_readwrite("DPC_threshold", &configuration::DPC_threshold)
        .def_readwrite("Bayer_orientation", &configuration::Bayer_orientation)
        .def_readwrite("Black_Level", &configuration::Black_Level)
        .def_readwrite("LSC_gain_00", &configuration::LSC_gain_00)
        .def_readwrite("LSC_gain_01", &configuration::LSC_gain_01)
        .def_readwrite("LSC_gain_02", &configuration::LSC_gain_02)
        .def_readwrite("LSC_gain_03", &configuration::LSC_gain_03)
        .def_readwrite("Gamma_value", &configuration::Gamma_value);

    // 2. Bind your functions
    // Note: If LSC takes the struct as an argument, define it like this:
    m.def("pipeline", [](py::array_t<int> image, const configuration& cfg) {
        // Your logic here: Call your CUDA kernel with cfg.LSC_gain_00, etc.
    }, "Perform LSC using configuration object");
}