import clr
import utils
import gnn


def train(options):
    if options.dataset == 1:
        dataset = 'B-Dataset'
    else:
        dataset = 'F-Dataset'
    AllNode = pd.read_csv('./data/'+dataset+'/Allnode.csv', header=0,names=['id','name'])
    Adj = load_file_as_Adj_matrix('./data/'+dataset+'/Alledge.csv')
    features = pd.read_csv('./data/'+dataset+'/AllNodeAttribute.csv', header = None)
    features = features.iloc[:,1:]
    adj, train_features = load_data(Adj,features)

    # Training settings
    dropout = 0.02
    in_size = train_features.shape[1]  
    hi_size = 64 
    name = locals() 

    model = GCN(nfeat=in_size,nhid=hi_size,nclass= 64,dropout=dropout)
    model.train()
    output, Emdebding_train = model(train_features, adj)
    Emdebding_GCN = pd.DataFrame(Emdebding_train.detach().numpy())
    Emdebding_GCN.to_csv('./data/'+dataset+'/Emdebding_GCN.csv', header=None,index=False)

