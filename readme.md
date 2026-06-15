<p align="center">
  <img src="images\CISP - CUDA Image Signal Processor.png" width="900">
</p>

A High-Performance GPU-accelerated Cuda based Image Signal Processor(ISP) that transforms RAW Bayer data to high quality RGB output. This is a modular pipeline with tunable parameter for each operation. CISP explores the parallel nature of image processing algorithms.


# Features 
- GPU accelerated modules
- End-to-end Bayer RAW to RGB image conversion.
- Real-time parameter tuning through a graphical user interface.
- Support for High-bit Depth images
- Hybrid CUDA/C++ and Python architecture combining GPU-accelerated performance with the flexibility of a Python-based workflow.

# ISP Operations Integrated

## Bayer Domain
The raw output from the sensor of a camera is typically a 2d array of intensity values proportional to the amount of light captured by each pixel. This values are arranged in a pattern of red, green and blue colors called Bayer array. Green makes up 50% of the values as it is used as a luminance substitute( Human visual system is more sensitive to green).

### Defective pixel correction
Image sensors may have defective pixels in them (either always high, low or fixed). Defective pixel correction is technique used to remove such pixels from the output sensor values. The algorithm calculates average difference for a pixel to its neighbors and if that difference exceeds a threshold the pixel is replaced with an interpolation among its neighbors.

### Lens Shading Correction
The light falling on the senosr through the lens is not uniform everywhere. it tends to have a peak at the centre and decreases to the edges. This effect is called vignetting. Lens shading correction is an algorithm aiming to correct the effect.

### Black Level Correction
Black level also called dark current is the minimum value a sensor can record even when no light is allowed to pass. The dark is caused by a lot of factors, the thermal energy generates electrons inside the sensor and exites the pixels is one. The black level is subracted from the image to get the actual pixels values.

### Automatic White Balance Gain Adjustment (Gray world assumption)
Due to different sources of illumination whites in an image might not true white. To correct this irregularity automatic white balance is done. Gray world assumes that the average value of the image is gray and adapts red and blue channels using a gain.

### De-Mosaicing
The bayer domain images are not fit for human viewing. The process of generating 3-channel RGB images from bayer image is called De-Mosaicing. The algorithm used here is called hamilton adams - edge aware interpolation. This algorithm calculates directional gradients and choses the path with least change. The edge aware algorithm ensures continuity and reduces the interpolation artifacts. 

## RGB Domain
### Color Correction Matrix 
Color correction matrix is a transformation applied on images to achieve optimum colors. CCM ensures that an isp reproduces the correct color or the subject.

### Color Space Conversion to YCbCr
YCbCr color space is good for tone and color adjustments as it seperates Luminance and Chrominance.

### Gamma Correction
Human perception of brightness is nonlinear: we are more sensitive to differences in dark tones than in bright tones. A camera sensor, however, captures light in a physically linear space (signal proportional to scene radiance, within sensor limits). To prepare this linear data for display, we apply a gamma encoding (or tone curve) that maps linear intensity values into a perceptually more uniform space before storing or displaying as standard RGB (e.g., sRGB).

## YCbCr Domain
### Tone and Color Adjustments
- Brightness
- Saturation
- Hue
- Tint
- Contrast
- Vibrance

### Image Enhancement
### Bilateral Filter
A bilateral filter is an edge-preserving smoothing technique that reduces noise in an image by averaging nearby pixels based on both their spatial closeness and their similarity in intensity, so that only similar pixels contribute significantly to the result while edges are preserved.
### Joint Bilateral filter (Luma guided filtering - Chroma denoising)
Similar to bilateral filter but uses Luminance channel as a guide to filter chroma channels and preserve sharp edges. Luminance channels contain more of the edge information.
### Highboost filtering
Is an unsharp mask. It computes the gaussian filter output of the image and subtracts it from the original image to obtain edge information. It then adds the resultant mask to the original image with a gain factor.

### Gaussian Blur
A low pass filter which filters out high frequency edge information from the image. It uses a gaussian function to generate a kernel.

# Tech Stack

- Cuda
- C++
- Python
- TKinter and TTKBootStrap (For Ui)
- Pybind11
- Rawpy
- numpy
- opencv ( For basic file operations )

# Important Note
The isp.cu and isp-2.cu are 2 different implementation of the same pipeline. isp.cu expects a 1d numpy array and predefined configuration parameters and outputs 3 numpy arrays in R G B order. isp-2.cu expects 10 1d cupy pointers of type float and int and configuration data. isp-2.py outputs data into the 3 integer pointers passed. The isp-2.cu is used in the app2.py to reduce the memory allocation overhead happening with every execution of the pipeline.   

# Data Flow
<p align="center">
  <img src="images\1.png" width="900">
</p>


# User Interface
<p align="center">
  <img src="images\ui.jpg" width="900">
</p>


# How To Use

A video demonstration of the repository is uploaded online. access through this link :: https://youtu.be/R_uROm2E-l8?si=l8i2KtWSejb-nH7R


clone repository : 
```bash
git clone https://github.com/mjithujanardhanan/CISP---Cuda-ISP-Pipeline.git
cd CISP---Cuda-ISP-Pipeline
pip install -r requirements.txt
```
app.exe executes the ISP application

### supported Formats

```bash
    ".dng", ".nef", ".nrw",
    ".cr2", ".cr3", ".crw",
    ".arw", ".raf", ".orf",
    ".rw2", ".pef", ".CR2"
 ```

input the path to the raw image and press enter to load. The toogle button can be used to activate and deactivate each block. Once the parameters are set click the run button to activate the pipeline. You can also save the processed image by clicking save.


### Files:
- app.exe       :: application
- app.py        :: UI code
- ISP.cu        :: Pipeline Code
- Sample_input  :: contains a few raw images for testing
- src           :: contains individual kernels for each operation

# Future work

- Guided Filter
- histogram equilization
- image compression

# References

1. Gonzalez, R. C., & Woods, R. E. *Digital Image Processing* (4th ed.). Pearson, 2018.
2. Szeliski, R. *Computer Vision: Algorithms and Applications* (2nd ed.). Springer, 2022.
3. Nayar, S. *First Principles of Computer Vision Specialization*. Columbia University, Coursera. Available at: https://www.coursera.org/specializations/firstprinciplesofcomputervision  PS.... check this out... its gold
4.  Holm, J. *Image Sensors and Signal Processing for Digital Still Cameras*. CRC Press, 2011.
5. Langseth, R., Gaddam, V. R., Stensland, H. K., Griwodz, C., Halvorsen, P., & Johansen, D. 
   *An Experimental Evaluation of Debayering Algorithms on GPUs for Recording Panoramic Video in Real-Time*. 
   International Journal of Multimedia Data Engineering and Management (IJMDEM), 6(3), 1–16, 2015.
   DOI: 10.4018/IJMDEM.2015070101
6. Huang, B.-C., & Fuh, C.-S.
   *Image Pipeline Algorithms for Standard Mobile Imaging Architecture Sensors*.
   Proceedings of the 18th IPPR Conference on Computer Vision, Graphics and Image Processing (CVGIP), Taipei, Taiwan, 2005.
7. NVIDIA Corporation. *CUDA C++ Best Practices Guide*. NVIDIA Developer Documentation. Available at: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
8. NVIDIA Corporation. *CUDA C++ Programming Guide*. NVIDIA Developer Documentation. Available at: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
9. Kuo, K.-T. *openISP: Open Image Signal Processor*. GitHub repository. Available at: https://github.com/cruxopen/openISP
10. AI Plays. *AI Plays YouTube Channel*. YouTube. Available at: https://www.youtube.com/@AISpeedrunners