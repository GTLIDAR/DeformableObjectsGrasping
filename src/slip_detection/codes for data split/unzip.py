import os
import zipfile

def unzip_all_files(dir):
    zip_files = os.listdir(dir)
    for file in zip_files:
        if file[-3:] == "zip":
            new_file_name = file[:-4]
            with zipfile.ZipFile(file, 'r') as zip_ref:
                zip_ref.extractall(new_file_name)
        elif file[-2:] == "=0":
            new_file_name = file[:-5]
            with zipfile.ZipFile(file, 'r') as zip_ref:
                zip_ref.extractall(new_file_name)


if __name__ == "__main__":
    unzip_all_files("./")