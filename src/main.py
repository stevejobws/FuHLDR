import scipy.sparse as sp
import pandas as pd
import numpy as np

def Train_Test_Sample(options):
    if options.dataset == 1:
        dataset = 'B-Dataset'
    else:
        dataset = 'F-Dataset'
    print(dataset)
    DDAsNum = pd.read_csv('./data/'+dataset+'/DrDiNum.csv', header=None)
    Nindex = pd.read_csv('../data/'+dataset+'/RandomList.csv',header=None)
    for i in range(len(Nindex)):
        kk = []
        for j in range(10):
            if j !=i:
                kk.append(j)
        index = np.hstack([np.array(Nindex)[kk[0]],np.array(Nindex)[kk[1]],np.array(Nindex)[kk[2]],np.array(Nindex)[kk[3]],np.array(Nindex)[kk[4]],
                           np.array(Nindex)[kk[5]],np.array(Nindex)[kk[6]],np.array(Nindex)[kk[7]],np.array(Nindex)[kk[8]]])
        DDAs_train= pd.DataFrame(np.array(DDAsNum)[index])
        DDAs_train.to_csv('../data/'+dataset+'/train'+str(i)+'.csv', header=None,index=False)
        DDAs_train = pd.DataFrame(np.array(DDAsNum)[np.array(Nindex)[i]])
        DDAs_train.to_csv('../data/'+dataset+'/test'+str(i)+'.csv', header=None,index=False)
        print(i)
        
def main(options):

    if options.dataset == 1:
        dataset = 'B-Dataset'
    else:
        dataset = 'F-Dataset'
    print(dataset)

    creat_var = locals() 
    creat_var = locals() 
    Negative = pd.read_csv('../data/'+dataset+'/NegativeSample.csv',header=None) 
    Nindex = pd.read_csv('../data/'+dataset+'/RandomList.csv',header=None)
    Attribute = pd.read_csv('../data/'+dataset+'/Emdebding_GCN.csv', header=None)
    Embedding = pd.read_csv('../data/'+dataset+'/metapath2vec10.txt', sep=' ',header=None,skiprows=2)
    Embedding = Embedding.sort_values(0,ascending=True).dropna(axis=1) 
    Embedding.set_index([0], inplace=True)
    Negative[2] = Negative.apply(lambda x: 0 if x[0] < 0 else 0, axis=1)
    Train_Test_Sample(options)
    for i in range(10):
        train_data = pd.read_csv('../data/'+dataset+'/train'+str(i)+'.csv',header=None)
        train_data[2] = train_data.apply(lambda x: 1 if x[0] < 0 else 1, axis=1)
        kk = []
        for j in range(10):
            if j !=i:
                kk.append(j)
        index = np.hstack([np.array(Nindex)[kk[0]],np.array(Nindex)[kk[1]],np.array(Nindex)[kk[2]],np.array(Nindex)[kk[3]],np.array(Nindex)[kk[4]],
                           np.array(Nindex)[kk[5]],np.array(Nindex)[kk[6]],np.array(Nindex)[kk[7]],np.array(Nindex)[kk[8]]])
        result = train_data.append(pd.DataFrame(np.array(Negative)[index]))    
        labels_train = result[2]
        
        data_train_feature = pd.concat([pd.concat([Attribute.loc[result[0].values.tolist()],Embedding.loc[result[0].values.tolist()]],axis=1).reset_index(drop=True),
               pd.concat([Attribute.loc[result[1].values.tolist()],Embedding.loc[result[1].values.tolist()]],axis=1).reset_index(drop=True)],axis=1)

        data_train_feature.to_csv('../data/'+dataset+'/train_data_'+str(i)+'.csv',header=0,index=0)
        labels_train.to_csv('../data/'+dataset+'/train_labels_'+str(i)+'.csv',header=0,index=0)
        creat_var['data_train'+str(i)] = data_train_feature.values.tolist()
        creat_var['labels_train'+str(i)] = labels_train
        print(len(data_train_feature))
        del labels_train, result, data_train_feature
        test_data = pd.read_csv('../data/'+dataset+'/test'+str(i)+'.csv',header=None)
        test_data[2] = test_data.apply(lambda x: 1 if x[0] < 0 else 1, axis=1)
        result = test_data.append(pd.DataFrame(np.array(Negative)[np.array(Nindex)[i]]))    
        labels_test = result[2]
        
        data_test_feature = pd.concat([pd.concat([Attribute.loc[result[0].values.tolist()],Embedding.loc[result[0].values.tolist()]],axis=1).reset_index(drop=True),
               pd.concat([Attribute.loc[result[1].values.tolist()],Embedding.loc[result[1].values.tolist()]],axis=1).reset_index(drop=True)],axis=1)

        data_test_feature.to_csv('../data/'+dataset+'/test_data_'+str(i)+'.csv',header=0,index=0)
        labels_test.to_csv('../data/'+dataset+'/test_labels_'+str(i)+'.csv',header=0,index=0)
        creat_var['data_test'+str(i)] = data_test_feature.values.tolist()
        creat_var['labels_test'+str(i)] = labels_test
        print(len(data_test_feature))
        del train_data, test_data, labels_test, result, data_test_feature
        print(i)
    

if __name__ == '__main__':
    import optparse
    import sys
    parser = optparse.OptionParser(usage=main.__doc__)
    parser.add_option("-d", "--dataset", action='store',
                      dest='dataset', default=2, type='int',
                      help=('The dataset of cross-validation '
                            '(1: B-Dataset; 2: F-Dataset)'))
    options, args = parser.parse_args()
    print(options)
    from train import train
    print(options.dataset)
    train(options)
    sys.exit(main(options))
