

import glob
import numpy


def remove_subject(file):
    f = open(file, "r+")
    lines = f.readlines()

    for line in lines:
        if line != "Subject:":
            f.write(line)
    print (f)
    f.truncate()
    f.close()

filenames = sorted(glob.glob('test-mails/spmsgc*.txt'))

filenames2 = sorted(glob.glob('test-mails/*-*msg*.txt'))

print (len(filenames) + len(filenames2))

test_data_files = []

for f in filenames:
    test_data_files.append(f)
    remove_subject(f)
    #print (f)

for f in filenames2:
    test_data_files.append(f)
    remove_subject(f)
    #print (f)

print (len(test_data_files))

filenames3 = sorted(glob.glob('train-mails/spmsgc*.txt'))

filenames4 = sorted(glob.glob('train-mails/*-*msg*.txt'))

print (len(filenames3) + len(filenames4))

train_data_files = []

for f in filenames3:
    train_data_files.append(f)
    remove_subject(f)
    #print (f)

for f in filenames4:
    train_data_files.append(f)
    remove_subject(f)
    #print (f)

print (len(train_data_files))





