import os
import numpy as np
import h5py
from mfcc import load_audio_and_mfcc
import argparse

def add_keys(root_folder):
    """Finds all transcription files first, and create a dictionary where (key, value) = (file_name, transcription)

    Args:
        root_folder (string): folder to look for files, dictionary will contain files in all subfolders

    Returns:
        data_dict: dictionary of (file_name, transcription) pairs
    """
    data_dict = {}

    for folder_name, subfolders, files in os.walk(root_folder):
        for file_name in files:
            if file_name.lower().endswith('.txt'):
                file_path = os.path.join(folder_name, file_name)
                with open(file_path, 'r') as file:
                    for line in file:
                        parts = line.split(' ', 1)
                        key = parts[0]
                        value = parts[1].strip()
                        data_dict[key] = {'transcription': value, 'mfcc': []}

    return data_dict

def add_mfccs(root_folder, dictionary):
    """Adds computed MFCCs for the .flac files that correspond to the dictionary keys / database entries
    TODO: can be extended to other file types

    Args:
        root_folder (string): folder to look for files, dictionary will contain files in all subfolders
        dictionary ({string, string}): previously created dictionary of (file_name, transcription) pairs

    Returns:
        dictionary: updated dictionary with mfcc values added for all the keys
    """

    for folder_name, subfolders, files in os.walk(root_folder):
        for file_name in files:
            if file_name.lower().endswith('.flac'):
                file_path = os.path.join(folder_name, file_name)
                mfcc = load_audio_and_mfcc(file_path)

                # Add mfcc to dictionary
                key = os.path.splitext(file_name)[0]

                if key in dictionary:
                    dictionary[key]['mfcc'].append(mfcc)

    return dictionary


def create_new_database(root_folder, db_path):
    """Creates a dictionary from the files in all subfolders of root, 
    then creates a new database from all the values in the specified location

    Args:
        root_folder (_type_): _description_
    """
    data_dict = add_keys(root_folder)
    data_dict = add_mfccs(root_folder, data_dict)

    # Add complete dictionary as database entries
    with h5py.File(db_path, 'w') as hf:
        for key, values in data_dict.items():
            group = hf.create_group(key)
            group.create_dataset('transcription', data=values['transcription'])
            mfcc_array = np.array(values['mfcc'])
            group.create_dataset('mfcc', data=mfcc_array)

    print(f"Database created successfully at: {db_path}")
    print("Total files: ", len(data_dict.keys()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add files to HDF5 database.")

    parser.add_argument('--root_folder', type=str, help='Root folder containing data', required=True)
    parser.add_argument('--db_path', type=str, help='Path to the HDF5 database file', required=True)

    args = parser.parse_args()
    create_new_database(args.root_folder, args.db_path)