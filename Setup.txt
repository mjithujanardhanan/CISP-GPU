#Follow the instructions to run the app.

#run the following command

pip install -r requirements.txt

# cupy is also required got to https://pypi.org/project/cupy/ and find the version corresponsing to your cuda version and install it.
# you can find you cuda version by executing    "NVCC --version"    in the terminal


cmake -S . -B build
cmake --build build --config Release

python app2.py


""" ISP.cu is the earlier version which contains the older version of the isp function, but the older version can be reused 
in other programs as it is standalone and accepts a 1d flattened image as input and outputs R G B as 1d flattened arrays. ISP-2.py is a persistent 
version of the isp it accepts memory pointers on device as input and outputs through one of the buffers provided. It is specifically written for the application interface.
CMakeLists.txt should be modified if you intend to use the ISP.cu version. newer updated are only present in ISP-2.cu """