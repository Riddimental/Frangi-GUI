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
import frangi_tensor
from tkinter import Toplevel, filedialog, ttk
from PIL import Image, ImageTk
from RangeSlider.RangeSlider import RangeSliderH
import napari
from scipy.ndimage import zoom
from skimage.transform import resize

root = ctk.CTk()

# Window dimensions and centering
window_width = 850
window_height = 720
last_resize_time = 0
voxel_size = 1 #mm
min_voxel_size = 0
resize_interval = 1 / 60  # Tiempo mínimo entre ejecuciones en segundos (60 fps)
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
image_name = ""
x = (screen_width - window_width) // 2
y = (screen_height - window_height) // 2
root.geometry(f"{window_width}x{window_height}+{x}+{y}")
root.title("Frangi MRI")
# Make the window not resizable
root.minsize(window_width, window_height)

hVar1 = tk.DoubleVar()
hVar1.set(0.0)# left handle variable
hVar2 = tk.DoubleVar()
hVar2.set(10.0)# right handle variable

# Ruta del directorio "temp"
temp_directory = "temp"

# Verificar si el directorio no existe
if not os.path.exists(temp_directory):
    # Crear el directorio "temp" si no existe
    os.makedirs(temp_directory)

# Defining global variables
max_value = 0
alpha_val = 0.5
beta_val=0.5
step_val = 1
sigmas = []
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

nii_2d_image = []
nii_file = []
nii_3d_image = []
nii_3d_image_original = []
slice_portion = 100
view_mode = ctk.StringVar(value="Axial")
selection_image = Image.new("RGB",(200,200),(0,0,0))

file_selected = False

# Make the window not resizable until file is selected
#root.resizable(False, False)

def close_program():
    filters.delete_temp()
    root.destroy()
    sys.exit()
    
root.protocol("WM_DELETE_WINDOW", close_program)

def open_napari():
    global nii_3d_image
    hold_button(view_3D_button)
    viewer = napari.Viewer()
    viewer.add_image(nii_3d_image, name='3D Image')
    viewer.dims.ndisplay = 3
    refresh_text("Opening napari viewer...")
    
    # Hide the layer controls
    viewer.window.qt_viewer.dockLayerControls.toggleViewAction().trigger()
    viewer.window.qt_viewer.dockLayerList.toggleViewAction().trigger()

    # Show the viewer
    viewer.window.show()
    release_button(view_3D_button)

def on_window_resize(event):
    global original_canvas_width, original_canvas_height, last_resize_time, file_selected
    
    current_time = time.time()
    if (current_time - last_resize_time < resize_interval) or not(file_selected):
        return
    
    last_resize_time = current_time
    
    original_canvas_width = canvas_frame.winfo_width()
    original_canvas_height = canvas_frame.winfo_height()
    try:
        refresh_image()
        # print('me movi a las ', current_time)
    except Exception as e:
        pass

# function to refresh the canva with the lates plot update
def refresh_image():
    global selection_image, scale_factor, original_canvas_height, original_canvas_width
    # giving the function time to avoid over-refreshing
    plot_image = Image.open("temp/plot.jpeg")
    
    # Get the dimensions of the canvas
    canvas_width = canvas_frame.winfo_width()
    canvas_height = canvas_frame.winfo_height()

    # Calculate the scaling factor to fit the canvas while preserving proportions
    scale_factor = min(canvas_width / plot_image.width, canvas_height / plot_image.height)

    # Calculate the new dimensions
    new_width = int(plot_image.width * scale_factor)
    new_height = int(plot_image.height * scale_factor)
    
    if new_height >= original_canvas_height:
        new_height = original_canvas_height
    if new_width >= original_canvas_width:
        new_width = original_canvas_width

    picture_canvas.config(width=new_width, height=new_height)
    # Resize the image
    plot_image = plot_image.resize((new_width, new_height))

    # Printing the picture in the canvas
    image = ImageTk.PhotoImage(plot_image)
    picture_canvas.image = image
    
    # Calculate the coordinates to center the image
    center_x = (canvas_width - new_width) / 2
    center_y = (canvas_height - new_height) / 2

    # adjusting the canvas to be the same size as the plot
    picture_canvas.config(width=plot_image.width, height=plot_image.height)

    # If the image exceeds canvas dimensions, adjust centering coordinates
    if center_x < 0:
        center_x = 0
    if center_y < 0:
        center_y = 0

    # Print the image centered within the canvas
    picture_canvas.create_image(0, 0, image=image, anchor="nw")
    

# function to plot the readed .nii image
def plot_image():
    global max_value, nii_2d_image, nii_3d_image, slice_portion, max_slice
    # selecting a slice out of the 3d image
    if(view_mode.get() == "Coronal"): # im the y axis "Coronal"
        max_slice = nii_3d_image.shape[1]
        if slice_portion >= nii_3d_image.shape[1]: slice_portion = (nii_3d_image.shape[1]-1)
        nii_2d_image = nii_3d_image[:,slice_portion,:]
    elif(view_mode.get() == "Axial"): # im the z axis "Axial"
        max_slice = nii_3d_image.shape[2]
        if slice_portion >= nii_3d_image.shape[2]: slice_portion = (nii_3d_image.shape[2]-1)
        nii_2d_image = nii_3d_image[:,:,slice_portion]
    elif(view_mode.get() == "Sagittal"): # im the x axis "Sagital"
        max_slice = nii_3d_image.shape[0]
        if slice_portion >= nii_3d_image.shape[0]: slice_portion = (nii_3d_image.shape[0]-1)
        nii_2d_image = nii_3d_image[slice_portion,:,:]
    
    # to find the range of the threshold slider
    max_value = nii_2d_image.max()
    
    # rotate the figure 90 degrees and resize
    nii_2d_image = np.rot90(nii_2d_image)
    nii_2d_image = cv2.resize(nii_2d_image, None, fx=2.4, fy=2.4)  # Resize to twice the size
    
    plt.imsave("temp/plot.jpeg", nii_2d_image, cmap='gray')
    #plt.close()
    root.after(15, refresh_image())
    
    
    #tk.Label(frangi_frame).pack(pady=0)
    segmentation_tabs.grid(row=3)
    frangi_frame.grid(row=4, padx=0)
    file_tools_frame.grid(row=5, pady=10, padx=20)
    apply_frangi_button.pack(pady=5)
    view_3D_button.pack(pady=5)
    save_file_button.pack(pady=5)
    slice_slider.configure(state="normal", to=max_slice)
    picture_canvas.pack(anchor="center", expand=True)
    view_dropdown.configure(state="normal")
    filters_button.configure(state="normal")
    restore_button.configure(state="normal")
    
button_text = ""

def hold_button(button):
    global button_text
    button_text = button.cget("text")
    button.configure(state="disabled", text="Loading...")
    
def release_button(button):
    global button_text
    button.configure(state="normal", text=button_text)
    button_text=""

def resample_image(image_data, target_voxel_size):
    current_voxel_sizes = np.array(image_data.header.get_zooms())
    shape = image_data.get_fdata().shape * (current_voxel_sizes/target_voxel_size)
    new_image_data = resize(image_data.get_fdata(), output_shape=shape, mode='constant')
    text = "image reshaped from ", image_data.get_fdata().shape , " to ", new_image_data.shape
    print(text)
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
    root.after(5000, update_text)
    

def add_image():
    global file_path, nii_3d_image_original, nii_3d_image, file_selected, voxel_size, original_canvas_width, original_canvas_height, min_voxel_size, image_name
    filters.delete_temp()
    # Get the dimensions of the canvas
    original_canvas_width = canvas_frame.winfo_width()
    original_canvas_height = canvas_frame.winfo_height()
    file_path = filedialog.askopenfilename(filetypes=[("NIfTI files", "*.gz *.nii")])
    if file_path:
        try:
            # reading file
            nii_file_original = nib.load(file_path)
            nii_file = nib.load(file_path).get_fdata()
            nii_file.shape

            # Access the header metadata
            header = nii_file_original.header
            # Get voxel sizes
            voxel_sizes = header.get_zooms()
            min_voxel_size = min(voxel_sizes)
            voxel_size = min_voxel_size
            print("Original Voxel Sizes:", voxel_sizes)
            #print("Minimum Voxel Size:", min_voxel_size)

            # Resample the image data to ensure cubic voxels
            if all(v == voxel_sizes[0] for v in voxel_sizes): #no need to resize
                nii_3d_image = nii_file
                print("no resample required")
            else:
                nii_file_resampled = filters.resample_image(nii_file_original, min_voxel_size)
                nii_3d_image = nii_file_resampled
            nii_3d_image_original = nii_3d_image

            noise_size = filters.voxel2mm(filters.calculate_noise(nii_3d_image), voxel_size) * 2
            # Apply correction factor to calculated sigma based on the paper's research
            corrected_calculated_sigma = noise_size - (0.223 * noise_size)
            # print("Aproximate Noise Size: ", noise_size," mm")
            print("Calculated Noise Sigma: ", corrected_calculated_sigma)
            # runs function to update background
            plot_image()
            #restore_original()
            file_selected = True
            root.resizable(True, True)
            text="Loaded Image: " + image_name
            refresh_text(text)
            print("Image loaded, Dim: ", nii_3d_image.shape)
            image_name = os.path.basename(file_path)

        except Exception as e:
            print("Error loading image:", e)
            refresh_text("Error loading image: " + str(e))
    else:
        print("No file selected")
        refresh_text("No file selected")


def apply_frangi():
    """
    Apply the Frangi filter to the input image based on the user's input parameters.
    """
    global nii_3d_image
    
    hold_button(apply_frangi_button)

    def apply_frangi_thread():
        global nii_3d_image  # Ensure that we modify the global variable
        
        if isbasic:
            # If the user has selected the basic option, apply the Frangi filter with a single scale value
            sigma = filters.mm2voxel(sigma_val, voxel_size) / 2
            sigmas = [sigma]
        else:
            # If the user has selected the advanced option, apply the Frangi filter with a range of scale values
            var1 = filters.mm2voxel(hVar1.get(), voxel_size) / 2
            var2 = filters.mm2voxel(hVar2.get(), voxel_size) / 2
            sigmas = np.linspace(var1, var2, step_val + 1)
        
        # Apply the Frangi filter to the input image
        #output = frangi_tensor.my_frangi_filter(nii_3d_image_original, sigmas, alpha_val, beta_val, isblack)
        output = frangi_tensor.my_frangi_filter_parallel(nii_3d_image_original, sigmas, alpha_val, beta_val, isblack)
        
        # Update the global variable with the result
        nii_3d_image = output
        release_button(apply_frangi_button)

        # Update the GUI using root.after to ensure it's done in the main thread
        root.after(0, refresh_text, "Frangi filter applied")
        root.after(0, plot_image)
        
    # Start the filtering operation in a new thread
    threading.Thread(target=apply_frangi_thread).start()

def save_file():
    global nii_3d_image
    # obtain the data
    hold_button(save_file_button)
    data = nii_3d_image
    '''# Define the percentage of the image dimensions to extract from the corner
    corner_percentage = 0.05  # 5% of the image dimensions

    # Calculate the size of the corner region
    corner_size = [int(dim * corner_percentage) for dim in data.shape]

    # Extract the corner sample (assuming the corner is at (0,0,0))
    corner_sample = data[:corner_size[0], :corner_size[1], :corner_size[2]]'''
    
    # Open dialog window to save the file
    file_path = filedialog.asksaveasfilename(defaultextension=".nii", filetypes=[("NIfTI files", "*.nii"), ("All files", "*.*")])
    
    # If the user cancels, the path will be empty
    if not file_path:
        release_button(save_file_button)
        return
    
    # Create a NIfTI From the data
    nii_file = nib.Nifti1Image(data, np.eye(4))  # listo mi rey, guarde el metadata aqui, graciassi es necesario
    nib.save(nii_file, file_path)
    
    '''
    # Function to add Rician noise to an image
    def add_rician_noise(image, sigma):
        noise = np.random.normal(loc=0, scale=sigma, size=image.shape)
        noisy_image = np.sqrt(image**2 + noise**2)
        return noisy_image

    # Function to create a noisy image with Rician noise
    def create_noisy_image(base_image, sigma):
        noisy_image = add_rician_noise(base_image, sigma)
        return noisy_image
    
    sigmas = np.linspace(0, 1 , 11)
    
    for i, sigma in enumerate(sigmas):
        noisy_image_1 = create_noisy_image(data, sigma)
        nifti_img_1 = nib.Nifti1Image(noisy_image_1, affine=np.eye(4))
        nib.save(nifti_img_1, f'ellipsoids_with_noise_sigma_{sigma:.1f}.nii.gz')
    '''
    
    refresh_text("File saved: " + file_path)
    release_button(save_file_button)

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
        black_vessels_switch.configure(text="Black")
        isblack = True
        print("Black Vessels True")
        refresh_text("Target: Black vessels")
    else:
        black_vessels_switch.configure(text="White")
        isblack = False
        print("Black Vessels False")
        refresh_text("Target: White vessels")

def switch_mode(event=None):
    global isbasic
    if isbasic:
        print("Advanced mode")
        #refresh_text("Advanced mode selected")
        isbasic = False
    else:
        print("Basic mode")
        #refresh_text("Basic mode selected")
        isbasic = True

def update_scale_range_label(*args):
    # Retrieve the values stored in the DoubleVar objects
    value1 = hVar1.get()
    value2 = hVar2.get()
    
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
    filters.gaussian_preview(nii_2d_image,filters.mm2voxel(sigma_val, min_voxel_size), root)
    text_val = "Diameter: {:.2f} mm".format(sigma_val)
    diameter_label.configure(text=text_val)
    
def change_scale_step(val):
    global step_val
    step_val = int(val)
    text_val = "Scale Step: {:}".format(step_val)
    step_value_label.configure(text=text_val)
 
def restore_original():
    global gaussian_intensity, hVar1, hVar2, beta_val, step_val, nii_3d_image, nii_3d_image_original, threshold_value
    picture_canvas.create_image(0, 0, image=picture_canvas.image, anchor="nw")
    gaussian_intensity = 0
    threshold_value = 0
    nii_3d_image = nii_3d_image_original
    refresh_text("Original image restored")
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
        filters.gaussian_preview(nii_2d_image,gaussian_intensity,root)
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
        threshold_value = int(val)
        filters.thresholding2d(nii_2d_image, threshold_value)
        text_val = "Threshold: " + str(threshold_value)
        label_Threshold.configure(text=text_val)
        refresh_image()
        
    def apply_threshold():
        global threshold_value, nii_3d_image
        nii_3d_image = filters.thresholding(nii_3d_image,threshold_value)
        plot_image()
        refresh_text("Threshold applied")
    
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
    filters_window.geometry("350x420")
    filters_window.resizable(True, False)
    # Set the maximum width of the window
    filters_window.maxsize(700, filters_window.winfo_screenheight())

    # spacer
    ctk.CTkLabel(master=filters_window,text="Filtering Options", height=40).pack(pady=15)
    
    # frame grid for filters
    filters_frame = ctk.CTkScrollableFrame(master=filters_window, width=300,height=230, orientation="horizontal")
    filters_frame.pack(fill="x", expand=True, padx=15)
    
    # Gaussian frame
    gaussian_frame = ctk.CTkFrame(master=filters_frame)
    gaussian_frame.grid(row=0, column=0, padx=15, pady=5)
    #gaussian_frame.pack()
    
    # Gaussian slider
    gaussian_label = ctk.CTkLabel(master=gaussian_frame, text="Gaussian Options", height=10)
    gaussian_label.pack(pady=15)

    # Label for the Gaussian slider
    text_val = "Gaussian intensity: " + str(gaussian_intensity)
    label_Gaussian = ctk.CTkLabel(master=gaussian_frame, text=text_val)
    label_Gaussian.pack()

    # Gaussian filter slider
    gaussian_slider = ctk.CTkSlider(master=gaussian_frame, from_=0, to=13 , command=change_gaussian_val, width=120)
    gaussian_slider.set(0)
    gaussian_slider.pack(pady=5)
    
    # Gaussian button
    gaussian_button = ctk.CTkButton(master=gaussian_frame, text="Apply Gaussian", command=apply_gaussian_3d, width=120)
    gaussian_button.pack(pady=5)
    
    # Scikit Frangi frame
    sci_frangi_frame = ctk.CTkFrame(master=filters_frame)
    sci_frangi_frame.grid(row=0, column=1, padx=15, pady=5)
    #gaussian_frame.pack()
    
    # Scikit Frangi slider
    sci_frangi_label = ctk.CTkLabel(master=sci_frangi_frame, text="Scikit Frangi Options", height=10)
    sci_frangi_label.pack(pady=15)
    
    # Scikit Frangi button
    gaussian_button = ctk.CTkButton(master=sci_frangi_frame, text="old Frangi", command=apply_sci_frangi, width=120)
    gaussian_button.pack(pady=5)
    
    # Thresholding frame
    thresholding_frame = ctk.CTkFrame(master=filters_frame)
    thresholding_frame.grid(row=0, column=2, padx=15, pady=5)
    
    # Thresholding options
    gaussian_label = ctk.CTkLabel(master=thresholding_frame, text="Thresholding Options", height=10)
    gaussian_label.pack(pady=15)

    # Label for the Thresholding slider
    text_val = "Threshold: " + str(threshold_value)
    label_Threshold = ctk.CTkLabel(master=thresholding_frame, text=text_val)
    label_Threshold.pack()

    # Threshold  slider
    threshold_slider = ctk.CTkSlider(master=thresholding_frame, from_=0, to=max_value, command=change_threshold_val, width=120)
    threshold_slider.set(0)
    threshold_slider.pack(pady=5)
    
    # An apply button for Threshold iterations
    threshold_apply_button = ctk.CTkButton(master=thresholding_frame, text ="Apply Threshold", command=apply_threshold)
    threshold_apply_button.pack(pady=5)
    
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
filters_button = ctk.CTkButton(tools_button_frame, state="disabled", text="More Filters", command=filters_window)
filters_button.pack(pady=5)


view_frame = tk.Frame(canva_tools_frame)
view_frame.grid(row=0,column=1,pady=10)

# Define the options for the dropdown
view_options = ["Axial", "Coronal", "Sagittal"]

def handle_dropdown_selection(selection):
    view_mode.set(selection)
    plot_image()    

# view mode label
title_label = ctk.CTkLabel(view_frame, text="View Mode:")
title_label.grid(row=0, padx=10, sticky="s")

# Create the custom dropdown
view_dropdown = ctk.CTkComboBox(master=view_frame, variable=view_mode, values=view_options, command=handle_dropdown_selection, state="disabled")
view_dropdown.grid(row=1, pady=10)

# slice frame
slice_frame = tk.Frame(canva_tools_frame)
slice_frame.grid(row=0,column=2,pady=5)

# Label for the slider
text_val = "Slice: " + str(slice_portion)
label_slice = ctk.CTkLabel(master=slice_frame, text=text_val)
label_slice.pack( padx=10)

max_slice = 200

# slider
slice_slider = ctk.CTkSlider(master=slice_frame, from_=1, to=max_slice,state="disabled", command=change_slice_portion, width=120)
slice_slider.set(slice_portion)
slice_slider.pack( padx=10)



# Logo
logo_frame = tk.Frame(master=left_frame_canvas)
logo_frame.grid(row=0, pady=15, padx=30)
canvas_logo = ctk.CTkCanvas(master=logo_frame, width=100, height=100)
canvas_logo.grid(row=0)
logo_image = tk.PhotoImage(file="img/logo.png").subsample(4, 4)
canvas_logo.create_image(50, 50, image=logo_image)

# Title of the tool under the logo
title_label = ctk.CTkLabel(logo_frame, text="MRI Frangi's \n Segmentation Tool")
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
text_sigma_val = "Diameter: {:.2f} mm".format(sigma_val)
diameter_label = ctk.CTkLabel(basic_tab, text=text_sigma_val, bg_color=message_label.cget('bg_color'))
diameter_label.pack(pady=5, padx=0)

diameter_slider = ctk.CTkSlider(basic_tab, from_=0.0, to=40.0, command=change_sigma_val, width=120)
diameter_slider.set(sigma_val)
diameter_slider.pack(padx=20, pady=2, fill='x')
diameter_slider.configure(bg_color=message_label.cget('bg_color'))
segmentation_tabs.add(basic_tab, text="Basic")
segmentation_tabs.bind("<<NotebookTabChanged>>", switch_mode)

#Advanced tab
advanced_tab = ttk.Frame(segmentation_tabs)

#scales range slider in advanced tab
text_val = "Target diameter: \n{:.2f} to {:.2f} mm".format(hVar1.get(), hVar2.get())
scale_range = ctk.CTkLabel(advanced_tab, text=text_val)
scale_range.pack(pady=(5,0))

# scale range slider
scale_range_slider = RangeSliderH(advanced_tab, [hVar1, hVar2], Width=130, Height=30, padX=5, min_val=0.0, bgColor=frangi_frame.cget('bg'), max_val=100, show_value=False, digit_precision='.1f', line_s_color='white', font_color='white',font_size=1, line_color='gray',bar_color_inner=frangi_frame.cget('bg'), bar_color_outer='gray')
scale_range_slider.pack(padx=5, pady=(0,2))

hVar1.trace_add("write", update_scale_range_label)
hVar2.trace_add("write", update_scale_range_label)

# scale step frame in advanced tab
scale_frame = tk.Frame(advanced_tab)
scale_frame.pack(pady=(0,5))

# scale step label
text_val = "Scale Step: {:}".format(step_val)
step_value_label = ctk.CTkLabel(scale_frame, text=text_val)
step_value_label.pack(pady=2, padx=0, anchor='w')

# scale step slider
max_val = 100
scale_steps_slider = ctk.CTkSlider(master=scale_frame, from_=1, to=max_val,number_of_steps=max_val-1, command=change_scale_step, width=100)
scale_steps_slider.set(step_val)
scale_steps_slider.pack(pady=(0, 2), padx=0, anchor='w')

segmentation_tabs.add(advanced_tab, text="Advanced")
segmentation_tabs.bind("<<NotebookTabChanged>>", switch_mode)

# alpha frame
alpha_frame = tk.Frame(frangi_frame)
alpha_frame.grid(row=2, column=1)

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
beta_frame.grid(row=3, column=1)

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
black_vessels_switch = ctk.CTkSwitch(frangi_frame,text="Black", variable=black_vessels, command=change_black_vessels)
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

alphaicon_label_left = tk.Label(frangi_frame, image=alphaicon_left)
alphaicon_label_left.grid(row=2, column=0, padx=10, sticky='s')  

alphaicon_label_right = tk.Label(frangi_frame, image=alphaicon_right)
alphaicon_label_right.grid(row=2, column=2, padx=10, sticky='s')  

betaicon_label_left = tk.Label(frangi_frame, image=betaicon_left)
betaicon_label_left.grid(row=3, column=0, padx=10, sticky='s')  

betaicon_label_right = tk.Label(frangi_frame, image=betaicon_right)
betaicon_label_right.grid(row=3, column=2, padx=10, sticky='s')  

# file tools frame
file_tools_frame = tk.Frame(left_frame_canvas)

# apply frangi button(left_frame_canvas).grid(row=4,pady=2)
apply_frangi_button = ctk.CTkButton(file_tools_frame,text="Apply Frangi",command=apply_frangi)

# 3D view button
view_3D_button = ctk.CTkButton(file_tools_frame,text="3D View",command=open_napari)

# Process image segmentation button
save_file_button = ctk.CTkButton(file_tools_frame, text="Save NIfTI", command=save_file)

root.bind("<Configure>", on_window_resize)

root.mainloop()