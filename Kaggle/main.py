import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re

def extract_csv_data(file, cols_to_clean = [], exclude = [[]]):
    #, exclude = [[]]
    data = pd.read_csv(file)

    for i, col in enumerate(cols_to_clean):
        exclude_pattern = re.compile('|'.join(exclude[i]))
        data = data[data[col].str.contains(exclude_pattern) == False]

    return data

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
'''
emails = extract_csv_data(
    'spam.csv',
    ['spam'],
    [['notes_inbox', 'discussion_threads']]
)
'''

def remove_duplicates(data):
    processed = set()
    result = []
    pattern = re.compile('X-FileName: .*')
    pattern2 = re.compile('X-FileName: .*?  ')

    for doc in data:
        doc = doc.replace('\n', ' ')
        doc = doc.replace(' .*?nsf', '')
        match = pattern.search(doc).group(0)
        match = re.sub(pattern2, '', match)

        if match not in processed:
            processed.add(match)
            result.append(match)

    return result

email_bodies = emails.message.as_matrix()
unique_emails = remove_duplicates(email_bodies)

print('There are a total of {} non-duplicate emails\n'.format(len(unique_emails)))
print('Sample email, unstructured content:\n\n', unique_emails[1000])