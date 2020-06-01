import csv
from os.path import dirname, abspath

# path = dirname(abspath(__file__)) + '/train_param.csv'
f = open('../base/train_param.csv', 'r')
with f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row['a'], row['b'], row['c'])
