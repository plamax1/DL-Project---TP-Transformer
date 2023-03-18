import glob
import os
import pathlib
path = pathlib.Path().resolve()
target_path = os.path.join(path, 'Dataset2/**/*.txt')
def get_test():

    for file in glob.iglob(target_path, recursive=True):
        data=[]
        with open(file) as f:
            lines = f.readlines()
            print('Loading file : ', file, ' file-len: ',  len(lines))

        stripped=[]
        for i in lines:
            stripped.append(i.strip())

        i=0
        while i<len(stripped)-1:
            data.append([stripped[i], stripped[i+1]])
            i=i+2

    data=pd.DataFrame(data,columns=['Question','Answer'])
    val_frac = 0.1 #precentage data in val
    val_split_idx = int(len(data)*val_frac) #index on which to split
    data_idx = list(range(len(data))) #create a list of ints till len of data
    np.random.shuffle(data_idx)

    #get indexes for validation and train
    val_idx, train_idx = data_idx[:val_split_idx], data_idx[val_split_idx:]
    print('len of train: ', len(train_idx))
    print('len of val: ', len(val_idx))

    #create the sets
    train = data.iloc[train_idx].reset_index().drop('index',axis=1)
    val = data.iloc[val_idx].reset_index().drop('index',axis=1)
    print("DataFrame Created")
    train_dataset = Train_Dataset(data, 'Question', 'Answer')
    return get_train_loader(train_dataset, batch_size)
