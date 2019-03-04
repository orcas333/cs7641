import numpy as np
import pandas as pd


#Stitch digits dataset into zipped csv

digits_test_images = pd.read_csv('digits_test_images.csv')
digits_train_images = pd.read_csv('digits_train_images.csv')
digits_test_labels = pd.read_csv('digits_test_labels.csv')
digits_train_labels = pd.read_csv('digits_train_labels.csv')

digits_images = pd.concat([digits_test_images,digits_train_images],0, ignore_index=True)
digits_labels = pd.concat([digits_test_labels,digits_train_labels],0, ignore_index=True)

digits_dataset = pd.concat([digits_images,digits_labels],1, ignore_index=True)
digits_dataset = digits_dataset.drop(digits_dataset.columns[0],axis=1)
digits_dataset = digits_dataset.convert_objects(convert_numeric=True)

print (digits_dataset)
digits_dataset.to_csv('out_digits.csv')


print (digits_dataset.shape)

# SPLIT_NUM = round(digits_dataset.shape[0] / 3 )
DIGITS_SPLIT_NUM = 10


digits_dataset_test, digits_dataset_train = digits_dataset.loc[:DIGITS_SPLIT_NUM,:], digits_dataset.loc[DIGITS_SPLIT_NUM:,:]
digits_dataset_test.to_csv('out_digits_test.csv')
digits_dataset_train.to_csv('out_digits_train.csv')

print (digits_dataset_test)