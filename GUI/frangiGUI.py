import getpass
import os
import sys
import threading
import time
import customtkinter as ctk
import cv2
import numpy as np
import tkinter as tk
import nibabel as nib
import matplotlib.pyplot as plt
import filters
import training
import tensor_frangi
from tkinter import Toplevel, filedialog, ttk
from PIL import Image, ImageTk
from RangeSlider.RangeSlider import RangeSliderH
import napari
from scipy.ndimage import zoom, distance_transform_edt
from skimage.transform import resize
import sounds
import SCALR
import faulthandler
import gc
faulthandler.enable()


root = ctk.CTk()

sounds.startup()
gc.collect()
# Function to show the window on top
def show_on_top():
    root.attributes("-topmost", True)
    root.update() 
    root.after(500, lambda: root.attributes("-topmost", False))

show_on_top()


# Window dimensions and centering
window_width = 850
window_height = 740
last_resize_time = 0
voxel_size = 1 #mm
min_voxel_size = 0
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
image_name = ""
x = (screen_width - window_width) // 2
y = (screen_height - window_height) // 2
root.geometry(f"{window_width}x{window_height}+{x}+{y}")
root.title("Frangi MRI")
root.minsize(window_width, window_height)

left_handle = tk.DoubleVar()
left_handle.set(0.0)
right_handle = tk.DoubleVar()
right_handle.set(5.0)

# Ruta del directorio "temp"
temp_directory = "temp"

# Verificar si el directorio no existe
if not os.path.exists(temp_directory):
    # Crear el directorio "temp" si no existe
    os.makedirs(temp_directory)

# Defining global variables
nii_file_original = None
nii_2d_image = []
nii_file = []
nii_3d_image = []
nii_pvs_output = []
nii_3d_image_original = []
mask = []
shell = []
sigmas = []
image_max_value = 0
calculated_noise = 0.0
calculated_snr = np.inf
alpha_val = 0.2
beta_val=0.8
step_val = 4
sigma_val = 1.5
isbasic = False
black_vessels = tk.IntVar()
black_vessels.set(1)
isblack = True
gaussian_intensity = 0
original_canvas_width = 0
original_canvas_height = 0
file_path = ""
threshold_value = 0
slice_portion = 100
view_mode = ctk.StringVar(value="Fix Z")
selection_image = Image.new("RGB",(200,200),(0,0,0))
file_selected = False

def close_program():
    filters.delete_temp()
    root.destroy()
    sys.exit()
    
root.protocol("WM_DELETE_WINDOW", close_program)

def open_napari():
    """
    Opens the MRI image in the Napari viewer without launching the rest of the application GUI.
    """
    global nii_3d_image
    hold_button(view_3D_button)
    viewer = napari.Viewer()
    viewer.add_image(nii_3d_image, name='3D Image')
    viewer.dims.ndisplay = 3
    refresh_text("Opening napari viewer...")
    viewer.window.qt_viewer.dockLayerControls.toggleViewAction().trigger()
    viewer.window.qt_viewer.dockLayerList.toggleViewAction().trigger()
    viewer.window.show()
    release_button(view_3D_button)

def on_window_resize(event):
    """
    Handles window resize events by refreshing the image in the canvas at a limited frame rate to optimize performance.
    """

    global original_canvas_width, original_canvas_height, last_resize_time, file_selected
    
    current_time = time.time()
    if (current_time - last_resize_time < 1/30) or not(file_selected):
        return
    
    last_resize_time = current_time
    
    original_canvas_width = canvas_frame.winfo_width()
    original_canvas_height = canvas_frame.winfo_height()
    try:
        refresh_image()
    except Exception as e:
        pass

# function to refresh the canva with the lates plot update
def refresh_image():
    """
    Refreshes and resizes the displayed image to fit within the canvas while preserving aspect ratio.
    Centers the image and updates the canvas dimensions accordingly to prevent distortion or overflow.
    """
    global selection_image, scale_factor, original_canvas_height, original_canvas_width
    plot_image = Image.open("temp/plot.jpeg")
    canvas_width = canvas_frame.winfo_width()
    canvas_height = canvas_frame.winfo_height()
    scale_factor = min(canvas_width / plot_image.width, canvas_height / plot_image.height)
    new_width = int(plot_image.width * scale_factor)
    new_height = int(plot_image.height * scale_factor)
    if new_height >= original_canvas_height:
        new_height = original_canvas_height
    if new_width >= original_canvas_width:
        new_width = original_canvas_width
    picture_canvas.config(width=new_width, height=new_height)
    plot_image = plot_image.resize((new_width, new_height))
    image = ImageTk.PhotoImage(plot_image)
    picture_canvas.image = image
    center_x = (canvas_width - new_width) / 2
    center_y = (canvas_height - new_height) / 2
    picture_canvas.config(width=plot_image.width, height=plot_image.height)
    if center_x < 0:
        center_x = 0
    if center_y < 0:
        center_y = 0
    picture_canvas.create_image(0, 0, image=image, anchor="nw")
    

def plot_image():
    """
    Extracts a 2D slice from the loaded 3D NIfTI image based on the selected view mode (Fix X, Y, or Z),
    rescales it for display, saves it as a temporary JPEG, and updates the GUI components accordingly.
    """
    global image_max_value, nii_2d_image, nii_3d_image, slice_portion, max_slice
    if(view_mode.get() == "Fix Y"):
        max_slice = nii_3d_image.shape[1]
        if slice_portion >= nii_3d_image.shape[1]: slice_portion = (nii_3d_image.shape[1]-1)
        nii_2d_image = nii_3d_image[:,slice_portion,:]
    elif(view_mode.get() == "Fix Z"):
        max_slice = nii_3d_image.shape[2]
        if slice_portion >= nii_3d_image.shape[2]: slice_portion = (nii_3d_image.shape[2]-1)
        nii_2d_image = nii_3d_image[:,:,slice_portion]
    elif(view_mode.get() == "Fix X"):
        max_slice = nii_3d_image.shape[0]
        if slice_portion >= nii_3d_image.shape[0]: slice_portion = (nii_3d_image.shape[0]-1)
        nii_2d_image = nii_3d_image[slice_portion,:,:]
    image_max_value = nii_2d_image.max()
    nii_2d_image = cv2.resize(nii_2d_image, None, fx=2.4, fy=2.4)
    plt.imsave("temp/plot.jpeg", nii_2d_image, cmap='gray')
    root.after(5, refresh_image())
    
def hold_button(button):
    button.configure(state="disabled")
    
def release_button(button):
    button.configure(state="normal")
    
def resample_image(image_data, target_voxel_size):
    """
    Resamples a NIfTI image to a specified target voxel size.

    Parameters:
        image_data (nibabel.Nifti1Image): The input image to be resampled.
        target_voxel_size (float or tuple): The desired voxel size for resampling.

    Returns:
        numpy.ndarray: The resampled image data as a NumPy array.
    """

    current_voxel_sizes = np.array(image_data.header.get_zooms())
    shape = image_data.get_fdata().shape * (current_voxel_sizes/target_voxel_size)
    new_image_data = resize(image_data.get_fdata(), output_shape=shape, mode='constant')
    text = "image reshaped from ", image_data.get_fdata().shape , " to ", new_image_data.shape
    print(text)
    print('='*100)
    refresh_text(text)
    return new_image_data

def refresh_text(text):
    def update_text():
        if not file_selected:
            default_text="Upload an image to begin"
        else:
            default_text="Loaded Image: " + image_name
        message_label.configure(text=default_text)

    if not file_selected:
        default_text="Upload an image to begin"
    else:
        default_text="Loaded Image: " + image_name
    message_label.configure(text=text)
    root.after(10000, update_text)
    
def measure_noise():
    global calculated_noise, predict_button, calculated_snr
    sounds.noise()
    noise_size, snr = filters.get_sigma_and_snr(nii_3d_image)
    calculated_noise = noise_size
    calculated_snr = snr
    print(f"Calculated Noise Sigma for {image_name}: ", calculated_noise," SNR: ", snr)
    print('='*100)
    refresh_text(f"Calculated Noise Sigma: {calculated_noise:.2f}, SNR: {snr:.2f}")
    predict_button.configure(state="normal", text="SCALR")
    

def add_image():
    global file_path, nii_3d_image_original, nii_3d_image, file_selected, voxel_size, original_canvas_width, original_canvas_height, min_voxel_size, image_name, nii_file_original, predict_button, slice_portion, black_vessels, nii_pvs_output
    filters.delete_temp()
    # Get the dimensions of the canvas
    original_canvas_width = canvas_frame.winfo_width()
    original_canvas_height = canvas_frame.winfo_height()
    file_path = filedialog.askopenfilename(filetypes=[("NIfTI files", "*.gz *.nii")])
    if file_path:
        try:
            # reading file
            predict_button.configure(state="disabled", text="Reading Noise")
            nii_file_original = nib.load(file_path)
            nii_file = nib.load(file_path).get_fdata()
            nii_file.shape
            #slice_portion = nii_file.shape[2]//2
            # Access the header metadata
            header = nii_file_original.header
            # Get voxel sizes
            voxel_sizes = header.get_zooms()
            min_voxel_size = min(voxel_sizes)
            voxel_size = min_voxel_size
            filters.set_min_voxel_size(min_voxel_size)
            print("Original Voxel Sizes:", voxel_sizes)
            print('='*100)

            # Resample the image data to ensure cubic voxels
            nii_file_original = filters.isometric_voxels(nii_file_original)
            nii_3d_image = nii_file_original.get_fdata()
            nii_pvs_output = np.zeros(nii_3d_image.shape)
            nii_3d_image_original = nii_3d_image
            plot_image()
            segmentation_tabs.grid(row=3)
            frangi_frame.grid(row=4, padx=0)
            file_tools_frame.grid(row=5, pady=10, padx=20)
            apply_frangi_button.pack(pady=5)
            view_3D_button.pack(pady=5)
            save_file_button.pack(pady=5)
            apply_overlay_button.pack(pady=5)
            slice_slider.configure(state="normal", to=max_slice)
            picture_canvas.pack(anchor="center", expand=True)
            filters_button.configure(state="normal")
            restore_button.configure(state="normal")
            # Run measure_noise in the background
            #threading.Thread(target=measure_noise).start()
            view_segmented_button.configure(state="normal")
            #restore_original()
            file_selected = True
            root.resizable(True, True)
            image_name = os.path.basename(file_path)
            apply_overlay_button.configure(text="Apply Overlay")
            text="Loaded Image: " + image_name
            refresh_text(text)
            measure_noise()
            print(f"Image loaded {image_name}, Dim: {nii_3d_image.shape}, Voxel Size: {voxel_size:.2f}mm, real sized dimensions {nii_3d_image.shape[0]*voxel_size:.2f}mm x {nii_3d_image.shape[1]*voxel_size:.2f}mm x {nii_3d_image.shape[2]*voxel_size:.2f}mm")
            print('='*100)
            print('='*100)

        except Exception as e:
            sounds.error()
            print("Error loading image:", e)
            print('='*100)
            refresh_text("Error loading image: " + str(e))
    else:
        print("No file selected")
        print('='*100)
        refresh_text("No file selected")


def apply_frangi():
    """
    Apply the Frangi filter to the input image based on the user's input parameters.
    """
    global nii_3d_image, nii_pvs_output
    
    hold_button(apply_frangi_button)

    def apply_frangi_thread():
        global nii_3d_image, nii_pvs_output  # Ensure that we modify the global variable
        
        if isbasic:
            # If the user has selected the modern option, apply the Frangi filter with a single scale value
            sigma = filters.mm2voxel(sigma_val) / 2
            sigmas = [sigma]
        else:
            # If the user has selected the classic option, apply the Frangi filter with a range of scale values
            var1 = filters.mm2voxel(left_handle.get()) / 2
            var2 = filters.mm2voxel(right_handle.get()) / 2
            sigmas = np.linspace(var1, var2, step_val + 1)
        
        # Apply the Frangi filter to the input image
        try:
            #output = tensor_frangi.my_frangi_filter_parallel_training(mod, sigmas, alpha_val, beta_val, isblack, mask, shell)
            output = tensor_frangi.my_frangi_filter_parallel(nii_3d_image_original, sigmas, alpha_val, beta_val, isblack)
            release_button(apply_frangi_button)
            # Update the global variable with the result
            nii_3d_image_normalized = (nii_3d_image - nii_3d_image.min()) / (nii_3d_image.max() - nii_3d_image.min())
            nii_pvs_output = output*nii_3d_image_normalized
            nii_3d_image = nii_pvs_output
            root.after(0, refresh_text, "Frangi filter applied")
        except Exception as error:
            sounds.error()
            print("Error applying Frangi filter:", error)
            print('='*100)
            refresh_text("Error applying Frangi filter")
            release_button(apply_frangi_button)
            root.after(0, refresh_text, "Frangi filter not applied")
        

        # Update the GUI using root.after to ensure it's done in the main thread
        root.after(0, plot_image)
        
    # Start the filtering operation in a new thread
    threading.Thread(target=apply_frangi_thread).start()
    #apply_frangi_thread()

def save_file():
    global nii_3d_image
    # obtain the data
    hold_button(save_file_button)
    data = nii_3d_image
    
    # Open dialog window to save the file
    file_path = filedialog.asksaveasfilename(defaultextension=".nii", filetypes=[("NIfTI files", "*.nii"), ("All files", "*.*")])
    
    # If the user cancels, the path will be empty
    if not file_path:
        release_button(save_file_button)
        return
    
    # Create a NIfTI From the data
    nii_file = nib.Nifti1Image(data, np.eye(4))  # listo mi rey, guarde el metadata aqui, graciassi es necesario
    nib.save(nii_file, file_path)
    
    
    refresh_text("File saved: " + file_path)
    release_button(save_file_button)

def create_noise():
    noisy_niftis_array = training.generate_noise(nii_file_original)
    for i, image in enumerate(noisy_niftis_array):
        noise, snr= filters.get_sigma_and_snr(image)
        image_name_clean = image_name.replace('.nii.gz', '')
        nib.save(image, f'{image_name_clean}_noise_{noise:.2f}.nii.gz')
    refresh_text("Noise created")

def save_mask():
    global mask
    mask = nii_3d_image_original
    print("Mask saved")
    print('='*100)

def create_shell():
    global shell, nii_3d_image
    # Get the original image (mask or object)
    tomask = nii_3d_image_original
    
    # Compute the distance transform of the inverted mask (background)
    #distance_transform = distance_transform_edt(tomask == 0)  # Now EDT of background
    
    outer_threshold = filters.mm2voxel(1.5)
    inner_threshold = filters.mm2voxel(0.1)
    
    # Create the outer shell: Select voxels within the outer threshold of the object's boundary
    #newshell = ((distance_transform > inner_threshold) & (distance_transform <= outer_threshold)).astype(float)
    
    #shell = newshell  # Update global shell
    #nii_3d_image = newshell  # visualize the shell
    nii_norm = nii_3d_image_original/nii_3d_image_original.max()
    nii_3d_image = nii_3d_image + nii_norm/700
    
    print("Outer shell created")
    print('='*100)
    
def prepare_masking():
    hold_button(prepare_masking_button)
    print("Preparing masking")
    print('='*100)
    save_mask()
    create_shell()
    print("Masking prepared")
    refresh_text("Masking prepared")
    print('='*100)
    release_button(prepare_masking_button)
    plot_image()
    
def change_slice_portion(val):
    global slice_portion
    slice_portion = int(val)
    text_val = "Slice: " + str(slice_portion)
    label_slice.configure(text=text_val)
    plot_image()
 
def change_alpha(val):
    global alpha_val
    alpha_val = float(val)
    text_val = "Alpha: {:.2f}".format(alpha_val)
    alpha_label.configure(text=text_val)

def change_black_vessels():
    global black_vessels, isblack
    if black_vessels.get() == 1:
        black_vessels_switch.configure(text="Dark")
        isblack = True
        print("Black Vessels True")
        print('='*100)
        refresh_text("Target: Black vessels")
    else:
        black_vessels_switch.configure(text="Bright")
        isblack = False
        print("Black Vessels False")
        print('='*100)
        refresh_text("Target: White vessels")

def switch_mode(event=None):
    global isbasic
    if isbasic:
        print("Classic mode")
        print('='*100)
        #refresh_text("Advanced mode selected")
        isbasic = False
    else:
        print("Modern mode")
        print('='*100)
        #refresh_text("Basic mode selected")
        isbasic = True

def update_scale_range_label(*args):
    # Retrieve the values stored in the DoubleVar objects
    value1 = left_handle.get()
    value2 = right_handle.get()
    
    # Format the values into the label text
    text_val = "Target diameter: \n{:.2f} to {:.2f} mm".format(value1, value2)
    
    # Update the label text
    scale_range.configure(text=text_val)
 
def change_beta(val):
    global beta_val
    beta_val = float(val)
    text_val = "Beta: {:.2f}".format(beta_val)
    vessel_length_label.configure(text=text_val)
    
def change_sigma_val(val):
    global sigma_val
    sigma_val = float(val)
    sigma_in_voxels = filters.mm2voxel(sigma_val)
    filters.gaussian_preview(nii_2d_image,sigma_in_voxels)
    text_val = "Target Diameter: \n{:.2f} mm".format(sigma_val)
    refresh_image()
    diameter_label.configure(text=text_val)
    
def predict_scale():
    hold_button(predict_button)
    #best_scale = regression_models.predict_scale(calculated_snr,min_voxel_size/2)
    best_scale = SCALR.predict(nii_file_original)
    change_sigma_val(2*best_scale) #the scale is the radius not the diameter
    diameter_slider.set(sigma_val)
    #diameter_slider.set(2*best_scale) #the scale is the radius not the diameter
    refresh_text(f"Scale predicted: {best_scale} mm")
    release_button(predict_button)
    
def change_scale_step(val):
    global step_val
    step_val = int(val)
    text_val = "Scale Step: {:}".format(step_val)
    step_value_label.configure(text=text_val)
    
pre_overlay = np.zeros_like(nii_3d_image)
def apply_overlay():
    global nii_3d_image, pre_overlay
    nii_norm = nii_3d_image_original/nii_3d_image_original.max()
    if apply_overlay_button.cget("text") == "Apply Overlay":
        pre_overlay = nii_3d_image
        nii_3d_image_norm = pre_overlay/pre_overlay.max()
        nii_3d_image = nii_3d_image_norm + nii_norm/7
        apply_overlay_button.configure(text="Remove Overlay")
        refresh_text("Overlay applied")
    else:
        nii_3d_image = pre_overlay
        apply_overlay_button.configure(text="Apply Overlay")
        refresh_text("Overlay removed")
    
    plot_image()
 
def restore_original():
    global gaussian_intensity, left_handle, right_handle, beta_val, step_val, nii_3d_image, nii_3d_image_original, threshold_value
    picture_canvas.create_image(0, 0, image=picture_canvas.image, anchor="nw")
    gaussian_intensity = 0
    threshold_value = 0
    nii_3d_image = nii_3d_image_original
    refresh_text("Original image restored")
    apply_overlay_button.configure(text="Apply Overlay")
    plot_image()     

def filters_window():
    
    global nii_2d_image, nii_3d_image
    
    ploted_image = Image.open('temp/plot.jpeg').convert('L')
    ploted_array = np.array(ploted_image)
    
    def restore_sliders():
        global threshold_value,gaussian_intensity,slice_portion
        gaussian_intensity=0
        gaussian_slider.set(0)
        threshold_value=0
        threshold_slider.set(0)

    def change_gaussian_val(val):
        global gaussian_intensity
        # Truncate intensity to ensure it's an integer and make it odd
        kernel_size = val
        gaussian_intensity = kernel_size
        filters.gaussian_preview(nii_2d_image,gaussian_intensity)
        text_val = "Gaussian Intensity: {:.1f}".format(gaussian_intensity)
        label_Gaussian.configure(text=text_val)
        refresh_image()
    
    def apply_gaussian_3d():
        global nii_3d_image, gaussian_intensity
        nii_3d_image = filters.gaussian3d(nii_3d_image,gaussian_intensity)
        refresh_text("Gaussian blur applied")
        plot_image()
    
    def apply_sci_frangi():
        global nii_3d_image
        nii_3d_image = filters.my_frangi_filter(nii_3d_image_original,sigmas, alpha_val, beta_val, isblack)
        plot_image()
        refresh_text("Scikit Frangi applied")
    
    def cancel_filter():
        plot_image()
        restore_sliders()
        filters_window.destroy()
        filters_button.configure(state="normal")

    def change_threshold_val(val):
        global threshold_value, nii_2d_image
        threshold_value = float(val)
        filters.thresholding2d(nii_2d_image, threshold_value)
        text_val = "Threshold: {:.2f}".format(threshold_value)
        label_Threshold.configure(text=text_val)
        refresh_image()
        
    def apply_threshold():
        global nii_3d_image
        nii_3d_image = filters.thresholding(nii_3d_image,threshold_value)
        plot_image()
        refresh_text("Threshold applied")
    
    def train_model():
        """
        Trains the SCALR model using a folder of MRI images and a corresponding PVS mask.

        All images in the selected folder must represent the same anatomical scan, possibly with different noise levels or resolutions. 
        A single PVS mask will be applied to all images. If only one MRI and one mask are available, place the MRI in a folder and select it accordingly.
        """

        global mri_images_folder, mask
        refresh_text("Upload MRI Folder")
        mri_images_folder = filedialog.askdirectory(title="Select MRI images folder")
        refresh_text("Upload PVS Mask image")
        mask_file = filedialog.askopenfilename(title="Select mask file", filetypes=[("NIfTI files", "*.gz *.nii")])
        mask = nib.load(mask_file)

        mri_images = [nib.load(os.path.join(mri_images_folder,f)) for f in os.listdir(mri_images_folder) if not f.startswith('.')]
        var1 = filters.mm2voxel(left_handle.get()) / 2
        var2 = filters.mm2voxel(right_handle.get()) / 2
        voxel_sigmas = np.linspace(var1, var2, step_val + 1)
        training.automate_mri_analysis(mri_images, mask, voxel_sigmas)
    
    # Toplevel object which will 
    # be treated as a new window
    filters_window = Toplevel(root)
    
    # deactivate the filters button while this windows is open to avoid repeated instances
    filters_button.configure(state="disabled")
    filters_window.protocol("WM_DELETE_WINDOW", cancel_filter)
    
    # sets the title of the
    # Toplevel widget
    filters_window.title("Image Filters Selector")
 
    # sets the geometry of toplevel
    filters_window_width = 390
    filters_window_height = 470
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - filters_window_width) // 2
    y = (screen_height - filters_window_height) // 2
    filters_window.geometry(f"{filters_window_width}x{filters_window_height}+{x}+{y}")
    filters_window.resizable(True, False)
    # Set the maximum width of the window
    filters_window.maxsize(700, filters_window.winfo_screenheight())
    filters_window.attributes("-topmost", True)

    # spacer
    ctk.CTkLabel(master=filters_window,text="Filtering Options", height=40).pack(pady=15)
    
    # Tabview for filter categories
    filters_tabs = ctk.CTkTabview(master=filters_window, width=320, height=250)
    filters_tabs.pack(padx=15, pady=10)

    # Add tabs
    filters_tabs.add("Gaussian")
    filters_tabs.add("Mask")
    filters_tabs.add("Frangi")
    filters_tabs.add("Thresholding")
    filters_tabs.add("Training")

    
    # Gaussian frame
    gaussian_frame = filters_tabs.tab("Gaussian")
    
    # Gaussian slider
    gaussian_label = ctk.CTkLabel(master=gaussian_frame, text="Gaussian Options", height=10)
    gaussian_label.pack(pady=15)

    # Label for the Gaussian slider
    text_val = "Gaussian intensity: " + str(gaussian_intensity)
    label_Gaussian = ctk.CTkLabel(master=gaussian_frame, text=text_val)
    label_Gaussian.pack()

    # Gaussian filter slider
    gaussian_slider = ctk.CTkSlider(master=gaussian_frame, from_=0, to=15, command=change_gaussian_val, width=120)
    gaussian_slider.set(0)
    gaussian_slider.pack(pady=5)
    
    # Gaussian button
    gaussian_button = ctk.CTkButton(master=gaussian_frame, text="Apply Gaussian", command=apply_gaussian_3d, width=120)
    gaussian_button.pack(pady=5)
    
    def load_mask():
        global nii_3d_image
        filters_window.attributes("-topmost", False)
        file_path = filedialog.askopenfilename(filetypes=[("NIfTI files", "*.gz *.nii")])
        if file_path:
            #save the file as a mask
            uploaded_mask = nib.load(file_path).get_fdata()
            shape = nii_3d_image.shape
            filters_window.attributes("-topmost", True)
            uploaded_mask[:,:int(shape[1]*0.13),:] = uploaded_mask[:,int(shape[1]*0.87):,:] = 0
            nii_3d_image = nii_3d_image * uploaded_mask
            print( "Mask loaded and applied" )
            print('='*100)
            cancel_filter()
            plot_image()
    
    # Load mask
    Load_mask_frame = filters_tabs.tab("Mask")

    #gaussian_frame.pack()
    
    # Load mask button
    load_mask_button = ctk.CTkButton(master=Load_mask_frame, text="Load Mask", command=load_mask, width=120)
    load_mask_button.pack(pady=5)
    
    # Scikit Frangi frame
    sci_frangi_frame = filters_tabs.tab("Frangi")
    #gaussian_frame.pack()
    
    # Scikit Frangi slider
    sci_frangi_label = ctk.CTkLabel(master=sci_frangi_frame, text="Scikit Frangi Options", height=10)
    sci_frangi_label.pack(pady=15)
    
    # Scikit Frangi button
    gaussian_button = ctk.CTkButton(master=sci_frangi_frame, text="Scikit Frangi", command=apply_sci_frangi, width=120)
    gaussian_button.pack(pady=5)
    
    # Thresholding frame
    thresholding_frame = filters_tabs.tab("Thresholding")
    
    # Thresholding options
    thresholding_label = ctk.CTkLabel(master=thresholding_frame, text="Thresholding Options", height=10)
    thresholding_label.pack(pady=15)

    # Label for the Thresholding slider
    text_val = "Threshold: " + str(threshold_value)
    label_Threshold = ctk.CTkLabel(master=thresholding_frame, text=text_val)
    label_Threshold.pack()

    # Threshold  slider
    threshold_slider = ctk.CTkSlider(master=thresholding_frame, from_=0, to=image_max_value, number_of_steps=30, command=change_threshold_val, width=120)
    threshold_slider.set(0)
    threshold_slider.pack(pady=5)
    
    # An apply button for Threshold iterations
    threshold_apply_button = ctk.CTkButton(master=thresholding_frame, text ="Apply Threshold", command=apply_threshold)
    threshold_apply_button.pack(pady=5)
    
    # Training frame
    training_frame = filters_tabs.tab("Training")
    
    # Traninig options
    training_label = ctk.CTkLabel(master=training_frame, text="Model Training", height=10)
    training_label.pack(pady=15)
    
    # An apply button for Threshold iterations
    training_button = ctk.CTkButton(master=training_frame, text ="Upload Files", command=train_model)
    training_button.pack(pady=5)
    
    buttons_frame = tk.Frame(master=filters_window)
    buttons_frame.pack(pady=25)
    
    # cancel Button
    cancel_filter_button = ctk.CTkButton(master=buttons_frame, text="Close",  command=cancel_filter)
    cancel_filter_button.grid(row=0, column=0, padx=5, pady=5)
    
    # restore filters button
    restore_button_filter = ctk.CTkButton(buttons_frame, text="Restore Original", command=restore_original)
    restore_button_filter.grid(row=0, column=1, padx=5, pady=5)


# left Frame which contains the tools and options
left_frame = ctk.CTkFrame(root, height=screen_height)
left_frame.pack(side='left', fill='y')

# Create a Canvas widget for left frame
left_frame_canvas = ctk.CTkFrame(left_frame, height=screen_height)
left_frame_canvas.pack(side='left', fill='both', expand=True)

# diferent Canvas overlaped on the rest of the window
# the main canvas frame
right_frame = ctk.CTkFrame(root, width=750, height=470)
right_frame.pack(pady=10, padx=10, fill="both", expand=True)

# frame that contains the canvas tools
canva_tools_frame = ctk.CTkFrame(right_frame)
canva_tools_frame.pack(side="top",pady=10, padx=10, fill="x")

messages_frame = ctk.CTkFrame(right_frame, height=20)
messages_frame.pack(side="bottom",pady=10, padx=10, fill="x")
message_label = ctk.CTkLabel(messages_frame)
refresh_text(f"Welcome {getpass.getuser()}")
message_label.pack()

# the main canvas frame
canvas_frame = ctk.CTkFrame(right_frame)
canvas_frame.pack(pady=10, padx=10,fill="both", expand=True)

# the picture canvas where we show the image (under the drawing canvas)
picture_canvas = ctk.CTkCanvas(canvas_frame)

canva_tools_frame.rowconfigure(1, weight=1)
canva_tools_frame.columnconfigure(0, weight=1)
canva_tools_frame.columnconfigure(1, weight=1)
canva_tools_frame.columnconfigure(2, weight=1)

# frame for buttons
tools_button_frame = tk.Frame(canva_tools_frame)
tools_button_frame.grid(row=0,column=0,pady=5)

# Clear canva button
restore_button = ctk.CTkButton(tools_button_frame, state="disabled", text="Restore Original", command=restore_original)
restore_button.pack(pady=5)

# Filters button
filters_button = ctk.CTkButton(tools_button_frame, state="disabled", text="More Options", command=filters_window)
filters_button.pack(pady=5)


view_frame = tk.Frame(canva_tools_frame)
view_frame.grid(row=0, column=1, pady=10, padx = 10, sticky="ew")

# Define the options for the segmented button
view_options = ["Fix X", "Fix Y", "Fix Z"]

def handle_segmented_button_selection(selection):
    view_mode.set(selection)
    plot_image()

# View mode label
title_label = ctk.CTkLabel(view_frame, text="View Mode:")
title_label.pack(padx=10)

# Create the segmented button
view_segmented_button = ctk.CTkSegmentedButton(
    master=view_frame,
    state="disabled",
    values=view_options,
    command=handle_segmented_button_selection
)
view_segmented_button.pack(padx=10, fill="x", expand=True)
# Set "Fix Z" as the default selection
view_segmented_button.set("Fix Z")

# slice frame
slice_frame = tk.Frame(canva_tools_frame)
slice_frame.grid(row=0,column=2,pady=5, sticky="ew")

# Label for the slider
text_val = "Slice: " + str(slice_portion)
label_slice = ctk.CTkLabel(master=slice_frame, text=text_val)
label_slice.pack( padx=10)

max_slice = 200

# slider
slice_slider = ctk.CTkSlider(master=slice_frame, from_=1, to=max_slice,state="disabled", command=change_slice_portion, width=120)
slice_slider.set(slice_portion)
slice_slider.pack(padx=10, fill="x", expand=True)



# Logo
logo_frame = tk.Frame(master=left_frame_canvas)
logo_frame.grid(row=0, pady=15, padx=30)
canvas_logo = ctk.CTkCanvas(master=logo_frame, width=100, height=100)
canvas_logo.grid(row=0)
logo_image = tk.PhotoImage(file="img/logo.png").subsample(4, 4)
canvas_logo.create_image(50, 50, image=logo_image)

# Title of the tool under the logo
title_label = ctk.CTkLabel(logo_frame, text="PVS \n Segmentation Tool")
title_label.grid(row=1, pady=5)

# Upload Button
upload_button = ctk.CTkButton(left_frame_canvas, text='Upload NIfTI File', command=add_image)
upload_button.grid(row=2,pady=10, padx=20)

tk.Label(left_frame_canvas).grid(row=4,pady=2)

# Frangi options frame
frangi_frame = tk.Frame(left_frame_canvas)

# scales frame
scales_frame = tk.Frame(frangi_frame)
scales_frame.grid(row=0, column=1)

#Frangi options tabs
segmentation_tabs = ttk.Notebook(master=left_frame_canvas)

#basic tab
basic_tab = ttk.Frame(segmentation_tabs)
basic_tab.columnconfigure(0, weight=1)

text_sigma_val = "Target Diameter: \n{:.2f} mm".format(sigma_val)
diameter_label = ctk.CTkLabel(basic_tab, text=text_sigma_val, bg_color=message_label.cget('bg_color'))
diameter_label.grid(row=0, column=0, sticky='nsew', padx=5, pady=(5,5))

diameter_slider = ctk.CTkSlider(basic_tab, from_=0.0, to=5.0, command=change_sigma_val, width=120)
diameter_slider.set(sigma_val)
diameter_slider.grid(row=1, column=0, sticky='nsew', padx=5, pady=5)

predict_button = ctk.CTkButton(basic_tab, text="Reading Noise",state="disabled", command=predict_scale)
predict_button.grid(row=2, column=0, sticky='nsew', padx=5, pady=10)

segmentation_tabs.add(basic_tab, text="Modern")
segmentation_tabs.bind("<<NotebookTabChanged>>", switch_mode)


#Advanced tab
advanced_tab = ttk.Frame(segmentation_tabs)

#scales range slider in advanced tab
text_val = "Target diameter: \n{:.2f} to {:.2f} mm".format(left_handle.get(), right_handle.get())
scale_range = ctk.CTkLabel(advanced_tab, text=text_val)
scale_range.pack(pady=(5,0))

# scale range slider
scale_range_slider = RangeSliderH(advanced_tab, [left_handle, right_handle], Width=130, Height=30, padX=5, min_val=0.0, bgColor=frangi_frame.cget('bg'), max_val=10.0, show_value=False, digit_precision='.1f', line_s_color='white', font_color='white',font_size=1, line_color='gray',bar_color_inner=frangi_frame.cget('bg'), bar_color_outer='gray')
scale_range_slider.pack(padx=5, pady=(0,2))

left_handle.trace_add("write", update_scale_range_label)
right_handle.trace_add("write", update_scale_range_label)

# scale step frame in advanced tab
scale_frame = tk.Frame(advanced_tab)
scale_frame.pack(pady=(0,5))

# scale step label
text_val = "Scale Step: {:}".format(step_val)
step_value_label = ctk.CTkLabel(scale_frame, text=text_val)
step_value_label.pack(pady=2, padx=0)

# scale step slider
max_val = 20
scale_steps_slider = ctk.CTkSlider(master=scale_frame, from_=1, to=max_val,number_of_steps=max_val-1, command=change_scale_step, width=100)
scale_steps_slider.set(step_val)
scale_steps_slider.pack(pady=(0, 2), padx=0)

segmentation_tabs.add(advanced_tab, text="Classic")
segmentation_tabs.bind("<<NotebookTabChanged>>", switch_mode)

# alpha frame
alpha_frame = tk.Frame(frangi_frame)
#alpha_frame.grid(row=2, column=1)

# alpha label
text_val = "Alpha: {:.2f}".format(alpha_val)
alpha_label = ctk.CTkLabel(alpha_frame, text=text_val)
alpha_label.pack()

# alpha slider
alpha_slider = ctk.CTkSlider(master=alpha_frame, from_=0.001, to=1,number_of_steps=300, command=change_alpha, width=100)
alpha_slider.set(alpha_val)
alpha_slider.pack()

# beta frame
beta_frame = tk.Frame(frangi_frame)
#beta_frame.grid(row=3, column=1)

# beta slider
text_val = "Beta: {:.2f}".format(beta_val)
vessel_length_label = ctk.CTkLabel(beta_frame, text=text_val)
vessel_length_label.pack()

beta_slider = ctk.CTkSlider(master=beta_frame, from_=0.001, to=1,number_of_steps=300, command=change_beta, width=100)
beta_slider.set(beta_val)
beta_slider.pack()

# label for the pen mode
switch_label = ctk.CTkLabel(frangi_frame, text="Vessels:")
switch_label.grid(row=4, column=1, pady=5)

# switch to mark background and foreground
black_vessels_switch = ctk.CTkSwitch(frangi_frame,text="Dark", variable=black_vessels, command=change_black_vessels)
black_vessels_switch.grid(row=5, column=1, pady=1)

#icons for sliders

rangeicon_left_path = "img/scale1.png"
rangeicon_right_path = "img/scale2.png"
rangeicon_left_image = Image.open(rangeicon_left_path)
rangeicon_right_image = Image.open(rangeicon_right_path)
rangeicon_left_image.thumbnail((15, 15))  
rangeicon_right_image.thumbnail((30, 30))  
rangeicon_left = ImageTk.PhotoImage(rangeicon_left_image)
rangeicon_right = ImageTk.PhotoImage(rangeicon_right_image)

alphaicon_right_path = "img/alpha2.png"
alphaicon_left_path = "img/alpha1.png"
alphaicon_left_image = Image.open(alphaicon_left_path)
alphaicon_right_image = Image.open(alphaicon_right_path)
alphaicon_left_image.thumbnail((30, 30))  
alphaicon_right_image.thumbnail((30, 30))  
alphaicon_left = ImageTk.PhotoImage(alphaicon_left_image)
alphaicon_right = ImageTk.PhotoImage(alphaicon_right_image)

betaicon_right_path = "img/beta2.png"
betaicon_left_path = "img/beta1.png"
betaicon_left_image = Image.open(betaicon_left_path)
betaicon_right_image = Image.open(betaicon_right_path)
betaicon_left_image.thumbnail((10, 10))  
betaicon_right_image.thumbnail((30, 30))  
betaicon_left = ImageTk.PhotoImage(betaicon_left_image)
betaicon_right = ImageTk.PhotoImage(betaicon_right_image)

c_icon_left_path = "img/c1.png"
c_icon_right_path = "img/c2.png"
c_icon_left_image = Image.open(c_icon_left_path)
c_icon_right_image = Image.open(c_icon_right_path)
c_icon_left_image.thumbnail((30, 30))  
c_icon_right_image.thumbnail((40, 40))  
c_icon_left = ImageTk.PhotoImage(c_icon_left_image)
c_icon_right = ImageTk.PhotoImage(c_icon_right_image)

alphaicon_label_left = tk.Label(frangi_frame, image=alphaicon_right)
#alphaicon_label_left.grid(row=2, column=0, padx=10, sticky='s')  

alphaicon_label_right = tk.Label(frangi_frame, image=alphaicon_left)
#alphaicon_label_right.grid(row=2, column=2, padx=10, sticky='s')  

betaicon_label_left = tk.Label(frangi_frame, image=betaicon_left)
#betaicon_label_left.grid(row=3, column=0, padx=10, sticky='s')  

betaicon_label_right = tk.Label(frangi_frame, image=betaicon_right)
#betaicon_label_right.grid(row=3, column=2, padx=10, sticky='s')  

# file tools frame
file_tools_frame = tk.Frame(left_frame_canvas)

# apply frangi button(left_frame_canvas).grid(row=4,pady=2)
apply_frangi_button = ctk.CTkButton(file_tools_frame,text="Apply Frangi",command=apply_frangi)

# 3D view button
view_3D_button = ctk.CTkButton(file_tools_frame,text="3D Render",command=open_napari)

# Process image segmentation button
save_file_button = ctk.CTkButton(file_tools_frame, text="Save NIfTI", command=save_file)
#save_mask_button = ctk.CTkButton(file_tools_frame, text="Save Mask", command=save_mask)
#create_shell_button = ctk.CTkButton(file_tools_frame, text="Create Shell", command=create_shell)
prepare_masking_button = ctk.CTkButton(file_tools_frame, text="Prepare Masking", command=prepare_masking)
#save_file_button = ctk.CTkButton(file_tools_frame, text="Create Noise", command=create_noise)
apply_overlay_button = ctk.CTkButton(file_tools_frame, text="Apply Overlay", command=apply_overlay)

root.bind("<Configure>", on_window_resize)

root.mainloop()