
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

__constant__ float d_CCM[9];

struct configuration
{
    std::vector<float> CCM;
    int length;
    int width;
};


//Kernel for applying color correction matrix
__global__ void CCM_Kernel(int* red, int* green, int* blue,  int width, int length)
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

        red[idx] =          (int)fmaxf(0, fminf(65535.0f, roundf(R)));
        green[idx] =        (int)fmaxf(0, fminf(65535.0f, roundf(G)));
        blue[idx] =         (int)fmaxf(0, fminf(65535.0f, roundf(B)));
    }
}

py::tuple Color_Correction_Matrix(py::array_t<int> Red, py::array_t<int> Green, py::array_t<int> Blue, const configuration& cfg )    
{
   
    auto buffer1 = Red.request();
    int *Red_Input = static_cast<int*>(buffer1.ptr);
    int array_size_1 = static_cast<int>(buffer1.size);

    auto buffer2 = Green.request();
    int *Green_Input = static_cast<int*>(buffer2.ptr);
    int array_size_2 = static_cast<int>(buffer2.size);

    auto buffer3 = Blue.request();
    int *Blue_Input = static_cast<int*>(buffer3.ptr);
    int array_size_3 = static_cast<int>(buffer3.size);

    int *D_Red, *D_Green, *D_Blue;  // array for storing rgb after debayering                                                                                                                                                

    cudaMalloc( &D_Red, array_size_1 * sizeof(int));   
    cudaMalloc( &D_Green, array_size_2 * sizeof(int)); 
    cudaMalloc( &D_Blue, array_size_3 * sizeof(int));   
                                                                                                                                     
    cudaMemcpy(D_Red , Red_Input, array_size_1 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(D_Green , Green_Input, array_size_2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(D_Blue , Blue_Input, array_size_3 * sizeof(int), cudaMemcpyHostToDevice);

    const int blockx= (cfg.width%16 == 0)?(cfg.width/16):(cfg.width/16 +1),blocky= (cfg.length%16 == 0)?(cfg.length/16):(cfg.length/16 +1);

    
    if(cfg.CCM.size() != 9)
    {
        throw std::runtime_error("CCM must contain exactly 9 floats");
    }

    cudaMemcpyToSymbol( d_CCM, cfg.CCM.data(), 9 * sizeof(float));


    CCM_Kernel<<<dim3(blockx,blocky),dim3(16,16)>>>(D_Red, D_Green, D_Blue, cfg.width, cfg.length);
    cudaDeviceSynchronize();

    cudaMemcpy(Red_Input, D_Red, array_size_1*sizeof(int), cudaMemcpyDeviceToHost);

    cudaMemcpy(Green_Input, D_Green, array_size_2*sizeof(int), cudaMemcpyDeviceToHost);

    cudaMemcpy(Blue_Input, D_Blue, array_size_3*sizeof(int), cudaMemcpyDeviceToHost);


    cudaFree(D_Red);
    cudaFree(D_Green);
    cudaFree(D_Blue);



    return py::make_tuple(Red, Green, Blue);

}

PYBIND11_MODULE(ccm, m) {
    // 1. Bind the configuration struct
    py::class_<configuration>(m, "Configuration")
        .def(py::init<>())
        .def_readwrite("CCM", &configuration::CCM)
        .def_readwrite("length", &configuration::length)
        .def_readwrite("width", &configuration::width);

    // 2. Bind your functions
    // Note: If LSC takes the struct as an argument, define it like this:
    m.def("Color_Correction_Matrix", &Color_Correction_Matrix, "Perform ccm using configuration object");
}