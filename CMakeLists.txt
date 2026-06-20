cmake_minimum_required(VERSION 3.18)

# 1. Define the project and enable CUDA and CXX (C++) languages
project(CudaMatrixAdd LANGUAGES CXX CUDA)

# 2. Set the C++ standard (pybind11 requires at least C++11, but 14 or 17 is better)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

# 3. Find pybind11
# This assumes you have pybind11 installed via pip or your package manager
execute_process(
    COMMAND python -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 PATHS ${pybind11_DIR} REQUIRED)

# 4. Define your module name (must match PYBIND11_MODULE name)
set(MODULE_NAME ISP)

# 5. Create the library
# pybind11_add_module handles the special flags needed for Python extensions
pybind11_add_module(${MODULE_NAME} ISP-2.cu)

# 6. Configure CUDA Architectures
# This tells NVCC which GPU generation to compile for. 
# 'native' detects your current GPU. For Qualcomm/Mobile, you'd specify a specific arch.
set_target_properties(${MODULE_NAME} PROPERTIES
    CUDA_ARCHITECTURES native
    CUDA_VISIBILITY_PRESET "hidden"
    CXX_VISIBILITY_PRESET "hidden"
)

set_target_properties(${MODULE_NAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/wrappers
    LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/wrappers

    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/wrappers
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/wrappers
)

# 7. Link CUDA libraries
# Even though we use the driver via <cuda_runtime.h>, we need to link it
target_link_libraries(${MODULE_NAME} PRIVATE pybind11::module)