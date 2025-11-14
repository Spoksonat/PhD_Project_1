import numpy as np
from tqdm import tqdm
import h5py
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import json
import pywt
from sklearn.cluster import KMeans
from Functions import *
import os

plt.rcParams['text.usetex'] = True

with open("../Config_files/scan_properties.json", "r") as f:
    data = json.load(f)
with open("../Config_files/dataset_groups.json", "r") as f:
    data_groups = json.load(f)

params = data["Function_parameters"]
dp = data["Data"] # dp stands for data properties

class Read_and_Process:
    def __init__(self, group_name, props=data, groups_data=data_groups):
        # Constructor: initializes attributes
        self.group_name = group_name
        self.props = props
        self.groups_data = groups_data

        self.params = self.props["Function_parameters"]
        self.dp = self.props["Data"] # dp stands for data properties

        self.path_data_results = dp["Data_folder_path"] + dp["Results_Data_path"]

        if(os.path.exists(self.path_data_results + f'{self.group_name}.h5')):
            print("File already exist")
        else:
            print("Creating File")
            with h5py.File(self.path_data_results + f'{self.group_name}.h5', 'w') as f_out:
                self.load_scan_groups(f_out)
                self.load_dataset_images(f_out)

    def save_dict_groups(self, group, upper_group, dict_obj):
        for key, value in dict_obj.items():
            if(key != "data"):
                dset = group.create_dataset(upper_group + "/" + key, data=value)
                dset.attrs[key] = dict_obj[key]

            else: # Assuming list of dictionaries
                for i, dict in enumerate(dict_obj[key]):
                    dgrp = group.create_group(upper_group + "/" + f"Run_{i+1}_Info")
                    for key2, value2 in dict.items():
                        dset2 = dgrp.create_dataset(key2, data=value2)
                        dset2.attrs[key2] = dict[key2]


    def load_scan_groups(self, f_out):
        initial_scan = self.groups_data[self.group_name]["Initial run"]
        final_scan = self.groups_data[self.group_name]["Final run"]
        ignored_scans = self.groups_data[self.group_name]["Ignored runs"]
    
        group_name = self.groups_data[self.group_name]["Logbook_scan_name"]
        suffix = self.groups_data[self.group_name]["Suffix group"]
    
        dataset_labels = [f"{group_name}_run{i}{suffix}" for i in range(initial_scan,final_scan+1)]
    
        for ignored_scan in ignored_scans:
            dataset_labels.remove(f"{group_name}_run{ignored_scan}{suffix}")
    
        sps = [data[dataset] for dataset in dataset_labels]
    
        dict_sps = {"Logbook group label": self.groups_data[self.group_name]["Logbook_scan_name"],
                    "LaTeX label": self.groups_data[self.group_name]["Scan_label"],
                    "Initial run": self.groups_data[self.group_name]["Initial run"],
                    "Final run": self.groups_data[self.group_name]["Final run"],
                    "Suffix": self.groups_data[self.group_name]["Suffix group"],
                    "Ignored runs": self.groups_data[self.group_name]["Ignored runs"],
                    "Ignored scans": self.groups_data[self.group_name]["Ignored scans"],
                    "data": sps}
        
        self.save_dict_groups(f_out, "Runs_Groups_Info", dict_sps)

        self.dict_sps = dict_sps

    def load_dataset_images(self, f_out):
        I0_normalization = bool(self.params["I0_normalization"])
        data_folder_path = self.dp["Data_folder_path"]
        sps = self.dict_sps["data"]
        ignored_scans = [f"{scan:0{5}d}" for scan in self.dict_sps["Ignored scans"]]
    
        scan_files_all = []
    
        for sp in sps:
            scan_files = np.arange(sp["Initial_scan"], sp["Final_scan"]+1, 1)
            scan_files_all.append(scan_files)
    
        scan_files_all = np.array(scan_files_all)
    
        E = np.empty(len(scan_files))
       

        for i in tqdm(range(len(scan_files))):
            scan_files_per_energy = scan_files_all[:,i]
            for j, scan in enumerate(scan_files_per_energy):
                if(f"{scan:0{5}d}" not in ignored_scans):
                    filename = f"Data/{define_dataset(scan)}/S{scan:0{5}d}/e20137_1_{scan:0{5}d}.h5" # Each number with up to     five zeroes to the left
                else:
                    scan = scan_files_per_energy[j-1]
                    filename = f"Data/{define_dataset(scan)}/S{scan:0{5}d}/e20137_1_{scan:0{5}d}.h5" # Each number with up to     five zeroes to the left
                with h5py.File(data_folder_path + filename, "r") as f:
                    I0 = f[dp["I0_key"]][:]
                    TFY_detector_images = f[self.dp["TFY_Eiger_key"]][:]
                    TFY_images_no_process = TFY_detector_images.copy()
                    if(I0_normalization == True):
                        TFY_detector_images = np.array([TFY_detector_images[k]/I0[k] for k in range(len(I0))])
                    TFY_detector_image = np.sum(TFY_detector_images, axis=0)
                    TFY_image_no_process = np.sum(TFY_images_no_process, axis=0)
                    TFY_detector_image = np.rot90(TFY_detector_image, -1)  
                    TFY_image_no_process = np.rot90(TFY_image_no_process, -1)
                    Transmission_image = np.sum(f[self.dp["Pilatus_key"]][:][:,:400, :250], axis=0)
    
                    key_I0 = "I0_Normalized_Images" + "/" + f"Scan_{j+1}" + "/" + f"Energy_bin_{i+1}_keV"
                    dset_I0 = f_out.create_dataset(key_I0, data=TFY_detector_image)
                        
                    key_non_I0 = "Non_Normalized_Images" + "/" + f"Scan_{j+1}" + "/" + f"Energy_bin_{i+1}_keV"
                    dset_non_I0 = f_out.create_dataset(key_non_I0, data=TFY_image_no_process)

                    key_T = "T_Images" + "/" + f"Scan_{j+1}" + "/" + f"Energy_bin_{i+1}_keV"
                    dset_T = f_out.create_dataset(key_T, data=Transmission_image)
                        
                    if(j==0):
                        E_scan = f[dp["Energy_key"]][0]
        
            E[i] = E_scan
    
        key_E = "Energy"
        dset_E = f_out.create_dataset(key_E, data=E)

    def scan_selection(self):

        self.file = h5py.File(self.path_data_results + f'{self.group_name}.h5', 'r')

        self.Energies = self.file["Energy"][...]

        considered_scans = self.groups_data[self.group_name]["Considered runs"]

        I0_images = []
        T_images = []

        for scan in tqdm(considered_scans):
            I0_per_run = []
            T_per_run = []
            for j in range(len(self.Energies)):
                I0_per_run.append(self.file[f"I0_Normalized_Images/Scan_{scan}/Energy_bin_{j+1}_keV"])
                T_per_run.append(self.file[f"T_Images/Scan_{scan}/Energy_bin_{j+1}_keV"])

            I0_images.append(np.array(I0_per_run))
            T_images.append(np.array(T_per_run))

        I0_images = np.array(I0_images)
        T_images = np.array(T_images)

        self.I0_normalized_TFY = np.sum(I0_images, axis = 0)
        self.T_images = np.sum(T_images, axis=0)

        self.cv_maps_per_E = np.std(I0_images, axis = 0)/np.mean(I0_images, axis = 0)

    def clean_outliers(self, thr_perc):

        self.I0_normalized_TFY = np.array([eliminate_outliers(image, low_perc=thr_perc[0], high_perc=thr_perc[1]) for image in self.I0_normalized_TFY])

        self.T_images = np.array([eliminate_outliers(img, low_perc=1, high_perc=99) for img in self.T_images])


    def calculate_T_profile(self):

        self.T_profile = np.sum(self.T_images, axis=(1,2))
        self.T_profile = self.T_profile/np.max(self.T_profile)

    def select_ROIs(self, ROIs):

        roi_signal = ROIs["Signal"]
        roi_bkg = ROIs["Bkg"]

        row_signal_start, row_signal_end = roi_signal[0].start, roi_signal[0].stop
        col_signal_start, col_signal_end = roi_signal[1].start, roi_signal[1].stop

        row_bkg_start, row_bkg_end = roi_bkg[0].start, roi_bkg[0].stop
        col_bkg_start, col_bkg_end = roi_bkg[1].start, roi_bkg[1].stop

        self.roi_signal = self.I0_normalized_TFY[:, row_signal_start:row_signal_end, col_signal_start:col_signal_end]
        self.roi_bkg = self.I0_normalized_TFY[:, row_bkg_start:row_bkg_end, col_bkg_start:col_bkg_end]       


    def calculate_XAS(self, ROIs,thr_perc = "Config"):

        if("Racemic" in self.group_name):
            T_norm = False
        else:
            T_norm = bool(self.params["T_normalization"])

        if(thr_perc == "Config"):
            thr_perc = (self.params["Low percentile TFY"], self.params["High percentile TFY"])

        self.scan_selection()
        self.clean_outliers(thr_perc)
        if(T_norm):
            self.calculate_T_profile()
        else:
            self.T_profile = 1
        self.select_ROIs(ROIs)

        n_signal = np.array([np.count_nonzero(self.roi_signal[i]) for i in range(len(self.roi_signal))])

        self.XAS = (FoM_XAS(self.roi_signal, n_signal))/self.T_profile
        self.XAS_bkg = (FoM_XAS(self.roi_bkg, n_signal))/self.T_profile
        self.XAS_std = np.zeros(len(self.XAS))
        self.XAS_bkg_std = np.zeros(len(self.XAS_bkg))

    def calculate_CV_map(self):
        self.cv_map = CV_map(self.I0_normalized_TFY)



        
        





        
        



            
    
            