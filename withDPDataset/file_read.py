
import glob
import numpy

#Ask question about email: 3-385msg2.txt

def remove_subject(file):
    f = open(file, "r+")
    lines = f.readlines()

    for line in lines:
        if line != "Subject:":
            f.write(line)
    print (f)
    f.truncate()
    f.close()

def changes_to_list(files, list):
    for f in files:
        list.append(f)
    return list

test_spam_files = sorted(glob.glob('spam-non-spam-dataset/test-mails/spmsgc*.txt'))
test_spam = []
test_spam = changes_to_list(test_spam_files, test_spam)
print (len(test_spam))

test_non_spam_files = sorted(glob.glob('spam-non-spam-dataset/test-mails/*-*msg*.txt'))
test_non_spam = []
test_non_spam = changes_to_list(test_non_spam_files, test_non_spam)
print (len(test_non_spam))

train_spam_files = sorted(glob.glob('spam-non-spam-dataset/train-mails/spmsgc*.txt'))
train_spam = []
train_spam = changes_to_list(train_spam_files, train_spam)
print (len(train_spam))

train_non_spam_files = sorted(glob.glob('spam-non-spam-dataset/train-mails/*-*msg*.txt'))
train_non_spam = []
train_non_spam = changes_to_list(train_non_spam_files, train_non_spam)
print (len(train_non_spam))







