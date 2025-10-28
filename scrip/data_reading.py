"""
Reading and Converting Calculation Results To Input

"""

import dpdata
from ase.io import write


def converting_input(cp2k_output_name_files,cp2k_output_name_files_fmt,cp2k_output_name_folder ,cp2k_output_name_fmt,deepmd_set_size ,output_files_name ):

    #reading the cp2k output data files
    data=dpdata.LabeledSystem('./',cp2k_output_name=cp2k_output_name_files,fmt=cp2k_output_name_files_fmt)

    #converting to npy data files for deepmd
    data.to_deepmd_npy(cp2k_output_name_folder ,fmt=cp2k_output_name_fmt,set_size=deepmd_set_size)

    #Convert to Allegro used data file
    ase_list = data.to("ase/structure")
    write("allegro_dataset.extxyz", ase_list)
