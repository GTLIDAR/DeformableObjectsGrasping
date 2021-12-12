import os
import re
import shutil
import sys

# For creating a custom dataset: it needs to contain three funcs: __init__, __len__, __getitem__
def Split_dataset(path):
    # Create train & test folders
    if not os.path.exists(os.path.join(path, 'training')):
        os.mkdir(os.path.join(path, 'training'))
    if not os.path.exists(os.path.join(path, 'testing')):
        os.mkdir(os.path.join(path, 'testing'))
    label_files = []
    train_data = []
    test_data = []
    for root, dirs, files in os.walk(path, topdown = True):
        for file in files:
            if file.endswith('.dat'):
                label_files.append(os.path.join(root, file))
    pat = re.compile(r'object([0-9]+)_result.dat')  # filter
    for label_file in label_files:
        idx = pat.search(label_file).group(1)
        test_label = ''
        train_label = ''
        fp = open(label_file, 'r')
        lines = fp.readlines()
        for i in range(len(lines)):
            line = lines[i]
            if i == 0 or i == len(lines) - 1:  #put in the test dataset
                test_label += line
                test_data.extend([line.replace('\n','') + ' ' + idx])
            else:
                train_label += line
                train_data.extend([line.replace('\n','') + ' ' + idx])
        tmp = open(os.path.join(path, "training", pat.search(label_file).group(0)), 'w')
        tmp.write(train_label)
        tmp.close()
        tmp = open(os.path.join(path, "testing", pat.search(label_file).group(0)), 'w')
        tmp.write(test_label)
        tmp.close()

    for train_data_piece in train_data:
        train_data_piece = train_data_piece.split(' ')
        source_folder_name = os.path.join(path, "object" + train_data_piece[-1], train_data_piece[-3] + "_mm")
        desti_folder_name = os.path.join(path, "training", "object" + train_data_piece[-1], train_data_piece[-3] + "_mm")
        if os.path.exists(source_folder_name):
            shutil.move(source_folder_name, desti_folder_name)

    for test_data_piece in test_data:
        test_data_piece = test_data_piece.split(' ')
        source_folder_name = os.path.join(path, "object" + test_data_piece[-1], test_data_piece[-3] + "_mm")
        desti_folder_name = os.path.join(path, "testing", "object" + test_data_piece[-1], test_data_piece[-3] + "_mm")
        if os.path.exists(source_folder_name):
            shutil.move(source_folder_name, desti_folder_name)

if __name__ == "__main__":
    Split_dataset(sys.argv[1])
