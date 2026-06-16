import cv2
import numpy as np
import tkinter as tk  
import ttkbootstrap as tb
from ttkbootstrap.constants import *
from ttkbootstrap.scrolled import ScrolledFrame
from PIL import Image, ImageTk
from wrappers import ISP
from pathlib import Path as Pt
import rawpy
import cupy as cp
import time

cp.cuda.Device(0).use()

RAW_EXTS = {
    ".dng", ".nef", ".nrw",
    ".cr2", ".cr3", ".crw",
    ".arw", ".raf", ".orf",
    ".rw2", ".pef", ".CR2",
    ".RAW"
}

cfg = ISP.Configuration()

cfg.white_level = 1024


class ISPPipelineUI(tb.Window):
    def __init__(self):
        super().__init__(themename="darkly") 
        self.title("CUDA ISP Pipeline Profiler")
        self.geometry("1200x750")
        self.raw_input_img = []
        self.count = 0
        self.arr_length =0
        self.white_balance = None
        self.black_level = None

        self.buf_1 = None
        self.buf_2 = None
        self.buf_3 = None
        self.buf_4 = None
        self.buf_5 = None
        self.buf_6 = None

        self.Red = None
        self.Green = None
        self.Blue = None





        self.orientation_list = {
            "BGGR" : 0, 
            "GBRG" : 1, 
            "GRBG" : 2, 
            "RGGB" : 3
        }

        # Main container
        self.main_frame = tb.Frame(self)
        self.main_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)

        # Text Input for Path
        tb.Label(self.main_frame, text="Path:").pack(fill=X, pady=(10, 0))
        self.path_input = tb.Entry(self.main_frame)
        self.path_input.insert(0, r"Sample_input\test.dng")
        self.path_input.pack(fill=X,padx = (0,130), pady=(0, 20))
        self.path_input.bind("<Return>", lambda event: self.Load_image(self.path_input.get()))

        # --- Left Panel: Scrollable Controls ---
        self.control_panel = ScrolledFrame(self.main_frame, width=320, autohide=True)
        self.control_panel.pack(side=LEFT, fill=Y, padx=(0, 15))

        # --- Drop-Down List (Combobox) for Orientation ---
        tb.Label(self.control_panel, text="Image Orientation:").pack(fill=X, pady=(10, 0))
        self.orientation_var = tb.StringVar(value="BGGR")
        self.orientation_dropdown = tb.Combobox(
            self.control_panel, 
            textvariable=self.orientation_var,
            values=["BGGR", "GBRG", "GRBG", "RGGB"],
            state="readonly", # 'readonly' prevents users from typing custom garbage text
            bootstyle=INFO
        )
        self.orientation_dropdown.pack(fill=X,padx=(0,20), pady=(0, 15))
        self.orientation_dropdown.bind("<<ComboboxSelected>>", lambda event: self.Initialize_parameters())

        # --- Pipeline Toggle Switches ---
        
        # Defective Pixel Correction
        self.dpc_var_toogle = tb.BooleanVar(value=True)
        self.dpc_cb = tb.Checkbutton(
            self.control_panel, text="Defective Pixel Correction", 
            variable=self.dpc_var_toogle, bootstyle="round-toggle"
        )
        self.dpc_cb.pack(fill=X, pady=10)

        # Black Level Correction
        self.blc_var_toggle = tb.BooleanVar(value=True)
        self.blc_cb = tb.Checkbutton(
            self.control_panel, text="Black Level Correction", 
            variable=self.blc_var_toggle, bootstyle="round-toggle"
        )
        self.blc_cb.pack(fill=X, pady=10)

        # Lens Shading Correction
        self.lsc_var_toggle = tb.BooleanVar(value=True)
        self.lsc_cb = tb.Checkbutton(
            self.control_panel, text="Lens Shading Correction", 
            variable=self.lsc_var_toggle, bootstyle="round-toggle"
        )
        self.lsc_cb.pack(fill=X, pady=10)

        # Automatic White Balance
        self.awb_var_toogle = tb.BooleanVar(value=True)
        self.awb_cb = tb.Checkbutton(
            self.control_panel, text="Automatic White Balance", 
            variable=self.awb_var_toogle, bootstyle="round-toggle",
            command=self.AWB_master_callback
        )
        self.awb_cb.pack(fill=X, pady=10)

        # Automatic White Balance - User defined
        self.awb_ud_var_toogle = tb.BooleanVar(value=False)
        self.awb_ud_cb = tb.Checkbutton(
            self.control_panel, text="User Defined", variable=self.awb_ud_var_toogle, 
            bootstyle="round-toggle", state=NORMAL
        )
        self.awb_ud_cb.pack(fill=X, padx=20, pady=5)

        # Exposure Compensation
        self.exp_var_toogle = tb.BooleanVar(value=True)
        self.exp_cb = tb.Checkbutton(
            self.control_panel, text="Exposure Compensation", 
            variable=self.exp_var_toogle, bootstyle="round-toggle"
        )
        self.exp_cb.pack(fill=X, pady=10)

        # Color Correction Matrix
        self.ccm_var_toogle = tb.BooleanVar(value=True)
        self.ccm_cb = tb.Checkbutton(
            self.control_panel, text="Color Correction Matrix", 
            variable=self.ccm_var_toogle, bootstyle="round-toggle"
        )
        self.ccm_cb.pack(fill=X, pady=10)

        # Color Space Conversion (Master Toggle)
        self.csc_var_toogle = tb.BooleanVar(value=False) # Switched to False initially to match sub-toggle DISABLED state
        self.csc_cb = tb.Checkbutton(
            self.control_panel, text="Color Space Conversion", 
            variable=self.csc_var_toogle, bootstyle="round-toggle", 
            command=self.csc_master_callback
        )
        self.csc_cb.pack(fill=X, pady=10)

        # --- Dependent Sub-Toggles  ---
        
        self.brightness_var_toogle = tb.BooleanVar(value=False)
        self.brightness_cb = tb.Checkbutton(
            self.control_panel, text="Brightness", variable=self.brightness_var_toogle, 
            bootstyle="round-toggle", state=DISABLED
        )
        self.brightness_cb.pack(fill=X, padx=20, pady=5)

        self.saturation_var_toogle = tb.BooleanVar(value=False)
        self.saturation_cb = tb.Checkbutton(
            self.control_panel, text="Saturation", variable=self.saturation_var_toogle, 
            bootstyle="round-toggle", state=DISABLED
        )
        self.saturation_cb.pack(fill=X, padx=20, pady=5)

        self.hue_var_toogle = tb.BooleanVar(value=False)
        self.hue_cb = tb.Checkbutton(
            self.control_panel, text="Hue", variable=self.hue_var_toogle, 
            bootstyle="round-toggle", state=DISABLED
        )
        self.hue_cb.pack(fill=X, padx=20, pady=5)

        self.contrast_var_toogle = tb.BooleanVar(value=False)
        self.contrast_cb = tb.Checkbutton(
            self.control_panel, text="Contrast", variable=self.contrast_var_toogle, 
            bootstyle="round-toggle", state=DISABLED
        )
        self.contrast_cb.pack(fill=X, padx=20, pady=5)

        self.tint_var_toogle = tb.BooleanVar(value=False)
        self.tint_cb = tb.Checkbutton(
            self.control_panel, text="Tint", variable=self.tint_var_toogle, 
            bootstyle="round-toggle", state=DISABLED
        )
        self.tint_cb.pack(fill=X, padx=20, pady=5)

        self.vibrance_var_toogle = tb.BooleanVar(value=False)
        self.vibrance_cb = tb.Checkbutton(
            self.control_panel, text="Vibrance", variable=self.vibrance_var_toogle, 
            bootstyle="round-toggle", state=DISABLED
        )
        self.vibrance_cb.pack(fill=X, padx=20, pady=5)

        #Bilateral Filter
        self.Bilateral_Filter_toogle = tb.BooleanVar(value=False)
        self.Bilateral_Filter = tb.Checkbutton(
            self.control_panel, text="Bilateral Filter", 
            variable=self.Bilateral_Filter_toogle, bootstyle="round-toggle", state=DISABLED, command= self.bilateral_master_callback
        )
        self.Bilateral_Filter.pack(fill=X, padx=20, pady=5)

        #Joint Bilateral Filter
        self.Joint_Bilateral_toogle = tb.BooleanVar(value=False)
        self.Joint_Bilateral = tb.Checkbutton(
            self.control_panel, text="Joint_Bilateral", 
            variable=self.Joint_Bilateral_toogle, bootstyle="round-toggle", state=DISABLED
        )
        self.Joint_Bilateral.pack(fill=X, padx=30, pady=5)

        #High Boost Filter
        self.Edge_enhancement_toogle = tb.BooleanVar(value=False)
        self.Edge_enhancement = tb.Checkbutton(
            self.control_panel, text="Edge enhancement", 
            variable=self.Edge_enhancement_toogle, bootstyle="round-toggle", state=DISABLED
        )
        self.Edge_enhancement.pack(fill=X, padx=20, pady=5)

        #Gaussian Blur Filter
        self.Gaussian_blur_toogle = tb.BooleanVar(value=False)
        self.Gaussian_blur = tb.Checkbutton(
            self.control_panel, text="Blur(Gaussian)", 
            variable=self.Gaussian_blur_toogle, bootstyle="round-toggle", state=DISABLED
        )
        self.Gaussian_blur.pack(fill=X, padx=20, pady=5)


        # Gamma Correction
        self.gamma_var_toogle = tb.BooleanVar(value=True)
        self.gamma_cb = tb.Checkbutton(
            self.control_panel, text="Gamma Correction", 
            variable=self.gamma_var_toogle, bootstyle="round-toggle"
        )
        self.gamma_cb.pack(fill=X, pady=10)


        

        # --- Entry Controls Area ---
        tb.Separator(self.control_panel, bootstyle="secondary").pack(fill=X, pady=15)

        # Float Input
        tb.Label(self.control_panel, text="White Level:").pack(fill=X, pady=(10, 0))
        self.WL_var = tb.DoubleVar(value=0.5)
        self.WL_input = tb.Entry(self.control_panel, textvariable=self.WL_var)
        self.WL_input.pack(fill=X, padx=(0,20), pady=(0, 10))
        self.WL_input.bind("<Return>") 
        
        # Slider
        tb.Label(self.control_panel, text="Defective Pixel Limit (0- 255 ):").pack(fill=X, pady=(10, 0))
        self.dpc_slider = tb.Scale(
            self.control_panel, from_=0.0, to=255.0, bootstyle=INFO
        )
        self.dpc_slider.set(50)
        self.dpc_slider.pack(fill=X, padx=(0,20), pady=(0, 10))                                                                  

        # --- Array Input: BLC Offset ---
        tb.Label(self.control_panel, text="BLC Offset (in bayer order):").pack(fill=X, pady=(10, 0))
        
        self.blc_offset_var = tb.StringVar(value=" 0 ,0 ,0 ,0")
        self.blc_array_input = tb.Entry(self.control_panel, textvariable=self.blc_offset_var)
        self.blc_array_input.pack(fill=X,padx=(0,20), pady=(0, 10))
        self.blc_array_input.bind("<Return>")

        # --- Array Input: LSC gains ---
        tb.Label(self.control_panel, text="Lens Shading correction radius Gains (0-1):").pack(fill=X, pady=(10, 0))
        
        self.lsc_offset_var = tb.StringVar(value=" 0.0 , 0.0, 0.0, 0.0")
        self.lsc_array_input = tb.Entry(self.control_panel, textvariable=self.lsc_offset_var)
        self.lsc_array_input.pack(fill=X,padx=(0,20), pady=(0, 10))
        self.lsc_array_input.bind("<Return>")

        # Float Input
        tb.Label(self.control_panel, text="Lens Shading correction radius (float):").pack(fill=X, pady=(10, 0))
        self.lsc_radius_var = tb.DoubleVar(value=1.0)
        self.lsc_radius_input = tb.Entry(self.control_panel, textvariable=self.lsc_radius_var)
        self.lsc_radius_input.pack(fill=X,padx=(0,20), pady=(0, 10))
        self.lsc_radius_input.bind("<Return>") 

        # --- Array Input: AWB gains ---
        tb.Label(self.control_panel, text=" White Balance Gains : ").pack(fill=X, pady=(10, 0))
        
        self.awb_gain_var = tb.StringVar(value=" 0.0 , 0.0, 0.0 ")
        self.awb_gain_input = tb.Entry(self.control_panel, textvariable=self.awb_gain_var)
        self.awb_gain_input.pack(fill=X,padx=(0,20), pady=(0, 10))
        self.awb_gain_input.bind("<Return>")

        # Exposure Compensation
        tb.Label(self.control_panel, text="Exposure compensation (Float):").pack(fill=X, pady=(10, 0))
        self.Exp_comp_var = tb.DoubleVar(value=0.5)
        self.Exp_comp_input = tb.Entry(self.control_panel, textvariable=self.Exp_comp_var)
        self.Exp_comp_input.pack(fill=X, padx=(0,20), pady=(0, 10))
        self.Exp_comp_input.bind("<Return>") 

        # --- Array Input: ccm  ---
        tb.Label(self.control_panel, text="Color Correction Matrix:").pack(fill=X, pady=(10, 0))
        
        self.color_correction_var = tb.StringVar(value="1.1, -0.05, -0.05, -0.05, 1.1, -0.05, -0.05, -0.05, 1.1")
        self.color_correction_array_input = tb.Entry(self.control_panel, textvariable=self.color_correction_var)
        self.color_correction_array_input.pack(fill=X, padx=(0,20), pady=(0, 10))
        self.color_correction_array_input.bind("<Return>")

        # Slider - brightness
        tb.Label(self.control_panel, text="Brightness:").pack(fill=X, pady=(10, 0))
        self.brightness_slider = tb.Scale(
            self.control_panel, from_=0.0, to=255.0, bootstyle=INFO
        )
        self.brightness_slider.set(0)
        self.brightness_slider.pack(fill=X, padx=(0,20), pady=(0, 10))

        # Slider - Saturation
        tb.Label(self.control_panel, text="Saturation:").pack(fill=X, pady=(10, 0))
        self.Saturation_slider = tb.Scale(
            self.control_panel, from_=0.0, to=3.0, bootstyle=INFO
        )
        self.Saturation_slider.set(1.0)
        self.Saturation_slider.pack(fill=X, padx=(0,20), pady=(0, 10))

        # Slider - Hue
        tb.Label(self.control_panel, text="Hue:").pack(fill=X, pady=(10, 0))
        self.Hue_slider = tb.Scale(
            self.control_panel, from_=0.0, to=(2 * np.pi), bootstyle=INFO
        )
        self.Hue_slider.set(0)
        self.Hue_slider.pack(fill=X, padx=(0,20), pady=(0, 10))

        # Slider - Contrast
        tb.Label(self.control_panel, text="Contrast:").pack(fill=X, pady=(10, 0))
        self.Contrast_slider = tb.Scale(
            self.control_panel, from_=1.0, to=1.4, bootstyle=INFO
        )
        self.Contrast_slider.set(0)
        self.Contrast_slider.pack(fill=X, padx=(0,20), pady=(0, 10))

        # Slider - Tint
        tb.Label(self.control_panel, text="Tint:").pack(fill=X, pady=(10, 0))
        self.Tint_slider = tb.Scale(
            self.control_panel, from_=0.0, to=100.0, bootstyle=INFO
        )
        self.Tint_slider.set(0)
        self.Tint_slider.pack(fill=X, padx=(0,20), pady=(0, 10))

        # Slider - Vibrance
        tb.Label(self.control_panel, text="Vibrance:").pack(fill=X, pady=(10, 0))
        self.Vibrance_slider = tb.Scale(
            self.control_panel, from_=1.0, to=3.0, bootstyle=INFO
        )
        self.Vibrance_slider.set(0)
        self.Vibrance_slider.pack(fill=X,padx=(0,20), pady=(0, 10))

        # Float Input
        tb.Label(self.control_panel, text="Bilateral filter - range STD (Float):").pack(fill=X, pady=(10, 0))
        self.RSTD_var = tb.DoubleVar(value=40)
        self.RSTD_input = tb.Entry(self.control_panel, textvariable=self.RSTD_var)
        self.RSTD_input.pack(fill=X, padx=(0,20), pady=(0, 10))
        self.RSTD_input.bind("<Return>") 


        # Float Input
        tb.Label(self.control_panel, text="Bilateral filter - domain STD (Float):").pack(fill=X, pady=(10, 0))
        self.DSTD_var = tb.DoubleVar(value=0.5)
        self.DSTD_input = tb.Entry(self.control_panel, textvariable=self.DSTD_var)
        self.DSTD_input.pack(fill=X,padx=(0,20), pady=(0, 10))
        self.DSTD_input.bind("<Return>") 

        # Integer Input
        tb.Label(self.control_panel, text="Bilateral Kernel Size (Int):").pack(fill=X, pady=(10, 0))
        self.Bilateral_kernel_var = tb.IntVar(value=3)
        self.Bilateral_kernel_input = tb.Spinbox(self.control_panel, from_=1, to=15, increment=2, textvariable=self.Bilateral_kernel_var)
        self.Bilateral_kernel_input.pack(fill=X,padx=(0,20), pady=(0, 10))
        self.Bilateral_kernel_input.bind("<Return>")

        # Float Input
        tb.Label(self.control_panel, text="Edge Enhancement - STD (Float):").pack(fill=X, pady=(10, 0))
        self.EESTD_var = tb.DoubleVar(value=0.5)
        self.EESTD_input = tb.Entry(self.control_panel, textvariable=self.EESTD_var)
        self.EESTD_input.pack(fill=X, padx=(0,20), pady=(0, 10))
        self.EESTD_input.bind("<Return>") 


        # Float Input
        tb.Label(self.control_panel, text="Edge Enhancement - Gain (Float):").pack(fill=X, pady=(10, 0))
        self.EGain_var = tb.DoubleVar(value=0.5)
        self.EGain_input = tb.Entry(self.control_panel, textvariable=self.EGain_var)
        self.EGain_input.pack(fill=X, padx=(0,20), pady=(0, 10))
        self.EGain_input.bind("<Return>") 


        # Integer Input
        tb.Label(self.control_panel, text="Edge Enhancement Kernel Size (Int):").pack(fill=X, pady=(10, 0))
        self.Edge_kernel_var = tb.IntVar(value=3)
        self.Edge_kernel_input = tb.Spinbox(self.control_panel, from_=1, to=15, increment=2, textvariable=self.Edge_kernel_var)
        self.Edge_kernel_input.pack(fill=X, padx=(0,20), pady=(0, 10))
        self.Edge_kernel_input.bind("<Return>")

        # Gaussian Filter
        tb.Label(self.control_panel, text="Gaussian- STD (Float):").pack(fill=X, pady=(10, 0))
        self.gaussian_var = tb.DoubleVar(value=0.5)
        self.gaussian_input = tb.Entry(self.control_panel, textvariable=self.gaussian_var)
        self.gaussian_input.pack(fill=X, padx=(0,20), pady=(0, 10))
        self.gaussian_input.bind("<Return>") 

        # Gaussian filter kernel size
        tb.Label(self.control_panel, text="Gaussian Filter Kernel Size (Int):").pack(fill=X, pady=(10, 0))
        self.gaussian_kernel_var = tb.IntVar(value=3)
        self.gaussian_kernel_input = tb.Spinbox(self.control_panel, from_=1, to=15, increment=2, textvariable=self.gaussian_kernel_var)
        self.gaussian_kernel_input.pack(fill=X, padx=(0,20), pady=(0, 10))
        self.gaussian_kernel_input.bind("<Return>")

        # Float Input
        tb.Label(self.control_panel, text="Gamma Correction (Float):").pack(fill=X, pady=(10, 0))
        self.Gamma_correction_var = tb.DoubleVar(value=2.2)
        self.Gamma_correction_input = tb.Entry(self.control_panel, textvariable=self.Gamma_correction_var)
        self.Gamma_correction_input.pack(fill=X, padx=(0,20), pady=(0, 10))
        self.Gamma_correction_input.bind("<Return>") 

        # Action Button
        self.process_btn = tb.Button(self.control_panel, text="Run Kernel", bootstyle=SUCCESS)
        self.process_btn.pack(fill=X, padx=(0,20), pady=(0, 20) )
        self.process_btn.bind("<Button-1>", lambda event: self.Action_button())

        # Action Button
        self.process_btn = tb.Button(self.control_panel, text="Save Image", bootstyle=SUCCESS)
        self.process_btn.pack(fill=X, padx=(0,20), pady=(0, 20) )
        self.process_btn.bind("<Button-1>", lambda event: self.save_image())

        # --- Right Panel: Image Display & Console ---
        self.image_frame = tb.Frame(self.main_frame, bootstyle="secondary")
        self.image_frame.pack(side=RIGHT, fill=BOTH, expand=True)

        # 1. Console Output Box (Packed BOTTOM)
        self.console_frame = tb.Frame(self.image_frame)
        self.console_frame.pack(side=BOTTOM, fill=X, pady=(10, 0))

        tb.Label(self.console_frame, text="Pipeline Logs:", font=("Helvetica", 10, "bold")).pack(anchor=NW)
        
        # Text widget for logs.
        self.console_box = tb.Text(self.console_frame, height=6, wrap=WORD)
        self.console_box.pack(fill=X, pady=(5, 0))
        self.console_box.insert(END, "ISP Profiler initialized...\n")
        self.console_box.configure(state=DISABLED) # Make it read-only for the user

        # 2. Image Label
        self.image_label = tb.Label(self.image_frame, text="Image will appear here", anchor=CENTER)
        self.image_label.pack(side=TOP, fill=BOTH, expand=True)

        self.tk_image = None 
        self.bind("<Configure>", self.on_resize)
        
        self.run_pipeline()


    def log_data(self, message):
        self.console_box.configure(state=NORMAL)       
        self.console_box.insert(END, f"{message}\n")   
        self.console_box.see(END)                      
        self.console_box.configure(state=DISABLED)     

    def bilateral_master_callback(self):
        if self.Bilateral_Filter_toogle.get():
            self.Joint_Bilateral.configure(state=NORMAL)
        else:
            self.Joint_Bilateral.configure(state=DISABLED)

    def csc_master_callback(self):
        self.update_sub_toggles()
        #self.run_pipeline()

    def AWB_master_callback(self):
        if self.awb_var_toogle.get():
            self.awb_ud_cb.configure(state=NORMAL)
        else:
            self.awb_ud_cb.configure(state=DISABLED)

    def update_sub_toggles(self):
        if self.csc_var_toogle.get():
            self.brightness_cb.configure(state=NORMAL)
            self.saturation_cb.configure(state=NORMAL)
            self.hue_cb.configure(state=NORMAL)
            self.contrast_cb.configure(state=NORMAL)
            self.tint_cb.configure(state=NORMAL)
            self.vibrance_cb.configure(state=NORMAL)
            self.Bilateral_Filter.configure(state=NORMAL)
            self.Edge_enhancement.configure(state=NORMAL)
            self.Gaussian_blur.configure(state=NORMAL)
        else:
            self.brightness_cb.configure(state=DISABLED)
            self.saturation_cb.configure(state=DISABLED)
            self.hue_cb.configure(state=DISABLED)
            self.contrast_cb.configure(state=DISABLED)
            self.tint_cb.configure(state=DISABLED)
            self.vibrance_cb.configure(state=DISABLED)
            self.Bilateral_Filter.configure(state=DISABLED)
            self.Edge_enhancement.configure(state=DISABLED)
            self.Joint_Bilateral.configure(state=DISABLED)
            self.Gaussian_blur.configure(state=DISABLED)

            

    def run_pipeline(self):
        if len(self.raw_input_img) == 0:
            return
        ISP.ISP(self.raw_input_img.data.ptr, self.buf_1.data.ptr, self.buf_2.data.ptr, self.buf_3.data.ptr, self.buf_4.data.ptr, self.buf_5.data.ptr, self.buf_6.data.ptr, self.Red.data.ptr, self.Green.data.ptr, self.Blue.data.ptr, cfg)
        

        self.log_data("image processed")

        R = self.Red.get()
        G = self.Green.get()
        B = self.Blue.get()

        R = np.reshape(R, (cfg.length, cfg.width))
        G = np.reshape(G, (cfg.length, cfg.width))
        B = np.reshape(B, (cfg.length, cfg.width))

        cur_image = np.dstack((R,G,B))
        cur_image = cur_image.astype( np.uint8)

        self.current_cv_img = cur_image
        self.display_image()

    def display_image(self):
        if not hasattr(self, 'current_cv_img'):
            return
        img_array = self.current_cv_img
        frame_w = cfg.width
        frame_h = cfg.length

        if frame_w < 10 or frame_h < 10:
            frame_w, frame_h = 800, 600 

        if len(img_array.shape) == 3:
            img_rgb = img_array
        else:
            img_rgb = img_array

        pil_img = Image.fromarray(img_rgb)
        img_w, img_h = cfg.width , cfg.length
        ratio = 0.25 
        new_w, new_h = 800, 600

        if new_w > 0 and new_h > 0:
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        self.tk_image = ImageTk.PhotoImage(pil_img)
        self.image_label.configure(image=self.tk_image, text="")

    def on_resize(self, event):
        if event.widget == self.image_frame or event.widget == self:
            self.display_image()
    
    def Load_image(self, Path):

        if Pt(Path).suffix in RAW_EXTS:
            with rawpy.imread(Path) as raw:
                loaded_img = np.array(raw.raw_image_visible.copy())
                cfg.white_level = raw.white_level
                self.WL_var.set(cfg.white_level)
                self.awb_gain_var.set( ", ".join(map(str, raw.camera_whitebalance[0:3])))
                self.blc_offset_var.set(", ".join(map(str, raw.black_level_per_channel)))
                self.color_correction_var.set(", ".join(map(str, raw.color_matrix[:,:3].flatten())))
        else:
            self.log_data(f"Error: Could not load image. Unsupported format")
            return
        
        self.log_data("initializing data")
        cfg.width = loaded_img.shape[1]
        cfg.length = loaded_img.shape[0]
        self.arr_length = cfg.length * cfg.width
        
        self.buf_1 = cp.empty( self.arr_length , dtype = cp.float32)
        self.buf_2 = cp.empty( self.arr_length , dtype = cp.float32)
        self.buf_3 = cp.empty( self.arr_length , dtype = cp.float32)
        self.buf_4 = cp.empty( self.arr_length , dtype = cp.float32)
        self.buf_5 = cp.empty( self.arr_length , dtype = cp.float32)
        self.buf_6 = cp.empty( self.arr_length , dtype = cp.float32)

        self.Red = cp.empty( self.arr_length , dtype = cp.int32)
        self.Green = cp.empty( self.arr_length , dtype = cp.int32)
        self.Blue = cp.empty( self.arr_length , dtype = cp.int32)
            
        self.log_data(f"Loaded successfully: {Path} | Shape: {cfg.width} , {cfg.length}")
        
        self.raw_input_img = cp.asarray(loaded_img, dtype = cp.float32).ravel()

        self.current_cv_img = (loaded_img /cfg.white_level)*255 

        self.log_data("Data Initializing complete")

        self.display_image()


    def Action_button(self):
        self.Initialize_parameters()
        self.run_pipeline()

    def Initialize_parameters(self):


        cfg.white_level = self.WL_var.get()

        cfg.orientation = self.orientation_list[self.orientation_var.get()]

        cfg.DPC = self.dpc_var_toogle.get()
        cfg.DPC_threshold =  self.dpc_slider.get()*cfg.white_level /255.0

        cfg.BLC = self.blc_var_toggle.get()
        blc_str = self.blc_offset_var.get()
        blc_list = [int(val.strip()) for val in blc_str.split(',')]
        if len(blc_list ) != 4  and cfg.BLC:
                self.log_data("Error: BLC Offset requires exactly 4 values.")
                return
        cfg.BLC_Offset = blc_list

        cfg.LSC = self.lsc_var_toggle.get()
        lsc_str = self.lsc_offset_var.get()
        lsc_list = [float(val.strip()) for val in lsc_str.split(',')]
        if len(lsc_list ) != 4  and cfg.LSC:
            self.log_data("Error: lsc gain requires exactly 4 values.")
            return
        cfg.LSC_gain = lsc_list
        cfg.LSC_Max_radius = self.lsc_radius_var.get()

        cfg.AWB = self.awb_var_toogle.get()
        cfg.AWB_Value_Given = self.awb_ud_var_toogle.get()
        awb_str = self.awb_gain_var.get()
        awb_list = [float(val.strip()) for val in awb_str.split(',')]
        if len(awb_list ) != 3  and cfg.AWB_Value_Given:
                self.log_data("Error: White balance requires exactly 4 values.")
                return
        cfg.AWB_gain = awb_list

        cfg.Exposure = self.exp_var_toogle.get()
        cfg.Exposure_value = self.Exp_comp_var.get()

        cfg.CCM = self.ccm_var_toogle.get()
        color_correction_str = self.color_correction_var.get()
        color_correction_list = [float(val.strip()) for val in color_correction_str.split(',')]
        if len(color_correction_list ) != 9 and cfg.CCM:
                self.log_data("Error: ccm requires exactly 9 values.")
                return
        cfg.CCM_gain = color_correction_list

        cfg.Color_Space_Conversion = self.csc_var_toogle.get()


        cfg.Brightness = self.brightness_var_toogle.get()
        cfg.Brightness_value = (self.brightness_slider.get() / 255.0) * cfg.white_level
 
        cfg.Saturation = self.saturation_var_toogle.get()
        cfg.Saturation_value = self.Saturation_slider.get()

        cfg.Hue = self.hue_var_toogle.get()
        cfg.Hue_value = self.Hue_slider.get()

        cfg.Contrast = self.contrast_var_toogle.get()
        cfg.Contrast_value = self.Contrast_slider.get()

        cfg.Tint = self.tint_var_toogle.get()
        cfg.Tint_value = self.Tint_slider.get()

        cfg.Vibrance = self.vibrance_var_toogle.get()
        cfg.Vibrance_value = self.Vibrance_slider.get()


        cfg.Bilateral_Filter = self.Bilateral_Filter_toogle.get()
        cfg.Joint_bilateral_kernel = self.Joint_Bilateral_toogle.get()
        cfg.Bilateral_spatial_STD = (self.DSTD_var.get()/255.0)* cfg.white_level
        cfg.Bilateral_Range_STD = self.RSTD_var.get()

        cfg.Bilateral_kernel_size = self.Bilateral_kernel_var.get()
        if cfg.Bilateral_kernel_size > 15:
            cfg.Bilateral_kernel_size = 15
            self.Bilateral_kernel_var.set(15)
        elif cfg.Bilateral_kernel_size < 1:
            cfg.Bilateral_kernel_size = 1
            self.Bilateral_kernel_var.set(1)
        elif cfg.Bilateral_kernel_size % 2 == 0:
            cfg.Bilateral_kernel_size = cfg.Bilateral_kernel_size -1
            self.Bilateral_kernel_var.set(cfg.Bilateral_kernel_size)

        cfg.Gaussian_blur = self.Gaussian_blur_toogle.get()
        cfg.Gaussian_STD = self.gaussian_var.get()
        cfg.Gaussian_blur_kernel_size = self.gaussian_kernel_var.get()
        if cfg.Gaussian_blur_kernel_size > 15:
            cfg.Gaussian_blur_kernel_size = 15
            self.gaussian_kernel_var.set(15)
        elif cfg.Gaussian_blur_kernel_size < 1:
            cfg.Gaussian_blur_kernel_size = 1
            self.gaussian_kernel_var.set(1)
        elif cfg.Gaussian_blur_kernel_size % 2 == 0:
            cfg.Gaussian_blur_kernel_size = cfg.Gaussian_blur_kernel_size -1
            self.gaussian_kernel_var.set(cfg.Gaussian_blur_kernel_size)

        cfg.Edge_enhancement = self.Edge_enhancement_toogle.get()
        cfg.Edge_enhancement_A_Value = self.EGain_var.get()
        cfg.Edge_enhancement_kernel_size = self.Edge_kernel_var.get()
        if cfg.Edge_enhancement_kernel_size > 15:
            cfg.Edge_enhancement_kernel_size = 15
            self.Edge_kernel_var.set(15)
        elif cfg.Edge_enhancement_kernel_size < 1:
            cfg.Edge_enhancement_kernel_size = 1
            self.Edge_kernel_var.set(1)
        elif cfg.Edge_enhancement_kernel_size % 2 == 0:
            cfg.Edge_enhancement_kernel_size = cfg.Edge_enhancement_kernel_size -1
            self.Edge_kernel_var.set(cfg.Edge_enhancement_kernel_size)

        cfg.Edge_enhancement_STD = self.EESTD_var.get()

        cfg.GAMMA = self.gamma_var_toogle.get()
        cfg.GAMMA_VALUE = self.Gamma_correction_var.get()
    def save_image(self):
        if not hasattr(self, 'current_cv_img'):
            self.log_data(" Image not loaded")
            return
        
        cv2.imwrite(f"output_{self.count}.png", cv2.cvtColor(self.current_cv_img,cv2.COLOR_BGR2RGB))
        self.log_data(" Image saved")
        self.count += 1


if __name__ == "__main__":
    app = ISPPipelineUI()
    app.mainloop()