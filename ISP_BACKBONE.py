import numpy as np
import matrix_add
import time
import matplotlib.pyplot as plt
from wrappers import ISP
import optuna

import cv2
import h5py
import rawpy

# with rawpy.imread(r"C:\Users\jithu\Downloads\test.RAW") as raw:
#     x = np.array(raw.raw_image.copy())

x = np.load("ppp.npy")

cfg = ISP.Configuration()

cfg.width = x.shape[1]
cfg.length = x.shape[0]
cfg.white_level = 4095
cfg.orientation = 0
cfg.DPC =True
cfg.DPC_threshold = 30
cfg.BLC = True
cfg.BLC_Offset = [155, 15, 15, 215]
cfg.LSC = False
cfg.LSC_gain = [1.0, 1.0, 1.0, 1.0]
cfg.LSC_Max_radius = 1.0
cfg.AWB = True
cfg.CCM = True
cfg.CCM_gain = [0.86,0.684,-0.079,0.367,0.428,0.394,-0.149,0.062,1.173] 
cfg.GAMMA = True
cfg.GAMMA_VALUE = 2.4
cfg.AWB_value = False
#cfg.AWB_gain = [3.808, 1.0, 2.323]
cfg.AWB_gain = [1.5, 1.0, 1.1]

x = x.flatten()


 


actual =cv2.imread(r"C:\Users\jithu\Downloads\a0001-jmac_DSC1459.tif")
actual = cv2.resize(actual, (cfg.width,cfg.length))
actual = cv2.cvtColor(actual, cv2.COLOR_BGR2RGB)


# def objective(trial):

#     # b1 = trial.suggest_int("b1", 0, 400 ,5 )
#     # b2 = trial.suggest_int("b2", 0, 400, 5)
#     # b3 = trial.suggest_int("b3", 0, 400, 5)
#     # cfg.BLC_Offset = [b1,b2,b2,b3]
#     # c1= trial.suggest_float("c1", 0.5, 1.2, step = 0.001)
#     # c2= trial.suggest_float("c2", 0.5, 1.0, step = 0.001)
#     # c3= trial.suggest_float("c3",-0.3, 0.3, step = 0.001)
#     # c4= trial.suggest_float("c4", -0.1, 0.5, step = 0.001)
#     # c5= trial.suggest_float("c5", 0.1, 0.8, step = 0.001)
#     # c6= trial.suggest_float("c6",0.0, 0.8, step = 0.001)
#     # c7= trial.suggest_float("c7", -0.5, 0.5, step = 0.001)
#     # c8= trial.suggest_float("c8", -0.25, 0.8, step = 0.001)
#     # c9= trial.suggest_float("c9", 1.0, 1.6, step = 0.001)
#     # cfg.CCM_gain = [ c1,c2,c3,c4,c5,c6,c7,c8,c9 ]
#     a1 = trial.suggest_int("a1", 10, 255 )
#     a2 = trial.suggest_int("a2", 5, 110 )


#     R, G, B = ISP.ISP(x,cfg)
#     image = np.dstack((R, G, B))
#     image = np.astype(image, np.uint8)
#     image = cv2.bilateralFilter(image, 5, a1, a2)
#     # R =np.astype(R, np.uint8)
#     # G =np.astype(G, np.uint8)
#     # B =np.astype(B, np.uint8)
#     # psnr_r = cv2.PSNR(R, actual[:, :, 0])
#     # psnr_g = cv2.PSNR(G, actual[:, :, 1])
#     # psnr_b = cv2.PSNR(B, actual[:, :, 2])

#     #return (psnr_r * 0.25 + psnr_g * 0.5 + psnr_b * 0.25)
#     return cv2.PSNR(image, actual)
    
# sampler=optuna.samplers.TPESampler(
#     seed=42
# )

# study = optuna.create_study(
#     direction="maximize",
#     sampler=sampler
# )

# # study.enqueue_trial({
# #     "c1": 0.86,
# #     "c2": 0.684,
# #     "c3": -0.079,
# #     "c4": 0.367,
# #     "c5": 0.428,
# #     "c6": 0.394,
# #     "c7":-0.149,
# #     "c8":0.062,
# #     "c9": 1.173
# # })

# study.optimize(
#     objective,
#     n_trials=5000,
#     show_progress_bar=True
# )

# print("\nBest PSNR:")
# print(study.best_value)

# print("\nBest Parameters:")
# for k, v in study.best_params.items():
#     print(f"{k}: {v}")

R, G, B = ISP.ISP(x,cfg)
image = np.dstack((R, G, B))
image = np.astype(image, np.uint8)

image =  cv2.bilateralFilter(image, 9, 44, 90)
print(cv2.PSNR(image, actual))
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)



cv2.imshow("Image", image)

cv2.waitKey(0)
cv2.destroyAllWindows()



