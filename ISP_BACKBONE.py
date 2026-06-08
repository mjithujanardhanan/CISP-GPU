import numpy as np
import time
import matplotlib.pyplot as plt
from wrappers import ISP
import optuna
import rawpy
import cv2
import h5py

cfg = ISP.Configuration()



with rawpy.imread(r"C:\Users\jithu\Desktop\git_repos\Differentiable-CUDA-based-ISP\Sample_input\IMG_7653.CR2") as raw:
    x = np.array(raw.raw_image_visible.copy())
    actual = np.array(raw.postprocess())
    cfg.white_level = raw.white_level

# with h5py.File(r"C:\Users\jithu\Desktop\git_repos\Differentiable-CUDA-based-ISP\Sample_input\0003_GT_RAW_010.MAT",'r') as f:
#     x = np.array(f['x'])

# x = x*1023



#x = np.load(r'Sample_input/input.npy')



cfg = ISP.Configuration()
cfg.width = x.shape[1]
cfg.length = x.shape[0]
cfg.orientation = 0
cfg.white_level = 4095

cfg.DPC =True
cfg.DPC_threshold = 30

cfg.BLC = True
cfg.BLC_Offset = [2047, 2048, 2047, 2047]

cfg.LSC = True
cfg.LSC_gain = [1.0, 1.0, 1.0, 1.0]
cfg.LSC_Max_radius = 1.0

cfg.AWB = True
cfg.AWB_Value_Given = False
cfg.AWB_gain = [1.8530800342559814, 0.9290269613265991, 1.3925764560699463]

cfg.CCM = True
cfg.CCM_gain = [1.2, -0.1, -0.1, -0.1,  1.2,  -0.1, -0.1, -0.1, 1.2]

cfg.Color_Space_Conversion = True 

cfg.Brightness = True
cfg.Brightness_value = 1.7

cfg.Saturation = True
cfg.Saturation_value = 1.3

cfg.Hue = True
cfg.Hue_value = np.pi /20

cfg.Contrast = True
cfg.Contrast_value = 1.1

cfg.Tint = True
cfg.Tint_value = 5

cfg.Vibrance = True
cfg.Vibrance_value = 0.35


cfg.Bilateral_Filter = True
cfg.Bilateral_Domain_STD = 1.5
cfg.Bilateral_Range_STD = 40
cfg.Bilateral_kernel_size =5


cfg.Edge_enhancement = True
cfg.Edge_enhancement_A_Value = 0.2
cfg.Edge_enhancement_kernel_size = 3
cfg.Edge_enhancement_STD = 0.2

cfg.GAMMA = False
cfg.GAMMA_VALUE = 2.4



x = x.flatten()



x1 = time.time()
R, G, B = ISP.ISP(x,cfg)
x2 = time.time()
print("Time taken for processing the image: ", x2-x1, "seconds")
image = np.dstack((R, G, B))
image = np.astype(image, np.uint8)


image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
image = cv2.resize(image, (int(cfg.width * 0.5), int(cfg.length * 0.5)))



img =image
scale = 1.0

def mouse(event, x, y, flags, param):
    global scale

    if event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            scale *= 1.1
        else:
            scale /= 1.1

cv2.namedWindow("Image")
cv2.setMouseCallback("Image", mouse)

while True:
    display = cv2.resize(img, None, fx=scale, fy=scale)
    cv2.imshow("Image", display)

    if cv2.waitKey(20) == 27:
        break

cv2.destroyAllWindows()



