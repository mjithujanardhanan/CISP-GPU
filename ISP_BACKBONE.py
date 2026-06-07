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



cfg = ISP.Configuration()
cfg.width = x.shape[1]
cfg.length = x.shape[0]
cfg.orientation = 3
cfg.DPC =True
cfg.DPC_threshold = 30
cfg.BLC = True
cfg.BLC_Offset = [2047, 2048, 2047, 2047]
cfg.LSC = False
cfg.LSC_gain = [1.0, 1.0, 1.0, 1.0]
cfg.LSC_Max_radius = 1.0
cfg.AWB = True
cfg.AWB_Value_Given = False
cfg.AWB_gain = [1.8530800342559814, 0.9290269613265991, 1.3925764560699463]
cfg.CCM = False
cfg.CCM_gain = [1.2, -0.1, -0.1, -0.1,  1.2,  -0.1, -0.1, -0.1, 1.2]
cfg.GAMMA = True
cfg.GAMMA_VALUE = 2.4
cfg.Color_Space_Conversion = True

cfg.Brightness = True
cfg.Brightness_value = 2.5
cfg.Saturation = True
cfg.Saturation_value = 1.2
cfg.Hue = False
cfg.Hue_value = np.pi/12
cfg.Contrast = True
cfg.Contrast_value = 1.01
cfg.Tint = True
cfg.Tint_value = 10
cfg.Vibrance = True
cfg.Vibrance_value = 2.0



cfg.Bilateral_Filter = True
cfg.Bilateral_Domain_STD = 2.5
cfg.Bilateral_Range_STD = 120.0
cfg.Bilateral_kernel_size =7
cfg.Edge_enhancement = True
cfg.Edge_enhancement_A_Value = 0.2
cfg.Edge_enhancement_kernel_size = 3
cfg.Edge_enhancement_STD = 0.2


x = x.flatten()


 


# actual =cv2.imread(r"C:\Users\jithu\Desktop\git_repos\Differentiable-CUDA-based-ISP\IMG_7653.JPG")
# actual = cv2.resize(actual, (cfg.width,cfg.length))
# actual = cv2.cvtColor(actual, cv2.COLOR_BGR2RGB)


# def objective(trial):

#     # b1 = trial.suggest_int("b1", 0, 400 ,5 )
#     # b2 = trial.suggest_int("b2", 0, 400, 5)
#     # b3 = trial.suggest_int("b3", 0, 400, 5)
#     # cfg.BLC_Offset = [b1,b2,b2,b3]
#     c1= trial.suggest_float("c1", 0.5, 1.2, step = 0.001)
#     c2= trial.suggest_float("c2", 0.5, 1.0, step = 0.001)
#     c3= trial.suggest_float("c3",-0.3, 0.3, step = 0.001)
#     c4= trial.suggest_float("c4", -0.1, 0.5, step = 0.001)
#     c5= trial.suggest_float("c5", 0.1, 0.8, step = 0.001)
#     c6= trial.suggest_float("c6",0.0, 0.8, step = 0.001)
#     c7= trial.suggest_float("c7", -0.5, 0.5, step = 0.001)
#     c8= trial.suggest_float("c8", -0.25, 0.8, step = 0.001)
#     c9= trial.suggest_float("c9", 1.0, 1.6, step = 0.001)
#     cfg.CCM_gain = [ c1,c2,c3,c4,c5,c6,c7,c8,c9 ]
#     # a1 = trial.suggest_int("a1", 10, 255 )
#     # a2 = trial.suggest_int("a2", 5, 110 )


#     R, G, B = ISP.ISP(x,cfg)
#     image = np.dstack((R, G, B))
#     image = np.astype(image, np.uint8)
#     # image = cv2.bilateralFilter(image, 5, a1, a2)
#     # R =np.astype(R, np.uint8)
#     # G =np.astype(G, np.uint8)
#     # B =np.astype(B, np.uint8)
#     # psnr_r = cv2.PSNR(R, actual[:, :, 0])
#     # psnr_g = cv2.PSNR(G, actual[:, :, 1])
#     # psnr_b = cv2.PSNR(B, actual[:, :, 2])

#     #return (psnr_r * 0.25 + psnr_g * 0.5 + psnr_b * 0.25)
#     return cv2.PSNR(image, actual)
    
# sampler=optuna.samplers.CmaEsSampler(
#     seed=42
# )

# study = optuna.create_study(
#     direction="maximize",
#     sampler=sampler
# )


# study.optimize(
#     objective,
#     n_trials=1000,
#     show_progress_bar=True
# )

# print("\nBest PSNR:")
# print(study.best_value)

# print("\nBest Parameters:")
# for k, v in study.best_params.items():
#     print(f"{k}: {v}")
x1 = time.time()
R, G, B = ISP.ISP(x,cfg)
x2 = time.time()
print("Time taken for processing the image: ", x2-x1, "seconds")
image = np.dstack((R, G, B))
image = np.astype(image, np.uint8)

image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
image = cv2.resize(image, (1920,1080))


cv2.imshow("Image", image)

cv2.waitKey(0)
cv2.destroyAllWindows()



