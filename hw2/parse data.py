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


# # # Madelon
madX1 = pd.read_csv('madelon_train_data.csv')
madX2 = pd.read_csv('madelon_valid_data.csv')
# madX = pd.concat([madX1,madX2],0)
madX = madX1
print (madX.shape)

madY1 = pd.read_csv('madelon_train_labels.csv')
madY2 = pd.read_csv('madelon_valid_labels.csv')
# madY = pd.concat([madY1,madY2],0)
madY = madY2

# madY.columns = ['Class']
mad = pd.concat([madX,madY],1)
mad = mad.dropna(axis=1,how='all')
mad.to_csv('test_mad_dataset.csv')


# print (mad)

MAD_SPLIT_NUM = 10

mad_dataset_test, mad_dataset_train = mad.loc[:MAD_SPLIT_NUM,:], digits_dataset.loc[MAD_SPLIT_NUM:,:]

mad_dataset_test.to_csv('out_mad_test.csv')
mad_dataset_train.to_csv('out_mad_train.csv')




# mad.to_hdf('datasets.hdf','madelon',complib='blosc',complevel=9)
#
#
#
#
# # Madelon
# madX1 = pd.read_csv('./madelon_train_data.csv',header=None,sep=' ')
# madX2 = pd.read_csv('./madelon_valid.data',header=None,sep=' ')
# madX = pd.concat([madX1,madX2],0).astype(float)
# madY1 = pd.read_csv('./madelon_train.labels',header=None,sep=' ')
# madY2 = pd.read_csv('./madelon_valid.labels',header=None,sep=' ')
# madY = pd.concat([madY1,madY2],0)
# madY.columns = ['Class']
# mad = pd.concat([madX,madY],1)
# mad = mad.dropna(axis=1,how='all')
# mad.to_hdf('datasets.hdf','madelon',complib='blosc',complevel=9)