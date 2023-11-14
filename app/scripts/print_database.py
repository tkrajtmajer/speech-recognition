import h5py
import argparse

def print_db(hdf5_filename='output.h5'):
    with h5py.File(hdf5_filename, 'r') as hdf5_file:
        # Print the keys at the root level of the HDF5 file
        print("Root keys:", list(hdf5_file.keys()))

        # Iterate through the keys and print the structure
        for key in hdf5_file.keys():
            print(f"\nGroup: {key}")
            group = hdf5_file[key]

            # Print attributes (metadata) of the group
            print("Attributes:")
            for attr_key, attr_value in group.attrs.items():
                print(f"  {attr_key}: {attr_value}")

            # Print datasets within the group
            print("Datasets:")
            for dataset_key in group.keys():
                dataset = group[dataset_key]

                # THIS if does not work, I cannot find a quick fix, too bad :(
                if dataset.dtype.kind == 'S':
                    values = dataset[()]
                    print(f"  {dataset_key}: {values}")
                else:
                    print(f"  {dataset_key}: {dataset.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add files to HDF5 database.")
    parser.add_argument('--db_path', type=str, help='Path to the HDF5 database file', required=True)

    args = parser.parse_args()
    print_db(args.db_path)