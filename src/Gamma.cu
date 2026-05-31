#include<cuda_runtime.h>
#include<stdio.h>
#include<cmath>
#include<stdlib.h>
#include<random>
#include<conio.h>
#include <algorithm>
#include<vector>
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>

namespace py=pybind11;

#define block_dim 16
#define block_size 256  

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

py::tuple Gamma_calculation(py::array_t<int> Red, py::array_t<int> Green, py::array_t<int> Blue, float Gamma, int width, int length)   // BGGR - 0,  GBRG -1, GRBG -2, RGGB -3 
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

    const int blockx= (width%16 == 0)?(width/16):(width/16 +1),blocky= (length%16 == 0)?(length/16):(length/16 +1);

    float Gamma_value = (1/Gamma);

    unsigned char *D_LUT;
    float x;
    cudaMalloc(&D_LUT, 65536 * sizeof(unsigned char));

    std::vector<unsigned char> LUT(65536);

    for(int i=0;i<65536;i++)
    {
        x = i/65535.0f;
        x = powf(x, (Gamma_value));

        LUT[i] = (int)roundf(x*255);
    }

    cudaMemcpy(D_LUT, LUT.data() , 65536 *  sizeof(unsigned char), cudaMemcpyHostToDevice);
    
    GAMMA_kernel<<<dim3(blockx,blocky), dim3(16,16)>>>(D_Red ,D_Green, D_Blue, D_LUT, width, length );

    cudaMemcpy(Red_Input, D_Red, array_size_1*sizeof(int), cudaMemcpyDeviceToHost);

    cudaMemcpy(Green_Input, D_Green, array_size_2*sizeof(int), cudaMemcpyDeviceToHost);

    cudaMemcpy(Blue_Input, D_Blue, array_size_3*sizeof(int), cudaMemcpyDeviceToHost);


    cudaFree(D_Red);
    cudaFree(D_Green);
    cudaFree(D_Blue);
    cudaFree(D_LUT);



    return py::make_tuple(Red, Green, Blue);



}

PYBIND11_MODULE(Gamma, m)
{
    m.def("Gamma_calculation", &Gamma_calculation, "perform gamma calculation on an image");
}