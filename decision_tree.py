from prediction import *
from dataset_details import *
from information_gain import *

# Load training data and assign column names
train,test = dataset_read("train path","test path")

#generate unique sequence to give unique id to each node of decision tree
def sequenceGenerator():
    global seqValue
    seqValue = seqValue + 1
    return seqValue

#recursively call this function to generate the tree
def grow_tree(dataset,nodeOwnNum,parent,child0,child1,depth1,depth2):

    global summary_train_model
    global tree_depth

    identical_feature = dataset.iloc[:,1:].drop_duplicates().shape[0]
    datasetClases = dataset.iloc[:,0].unique()

    #stop developing tree if dataset target variable has only one class or all the dataset points have same value
    if(len(datasetClases) == 1 or identical_feature == 1 or depth1 >= tree_depth or depth2 >= tree_depth):

        if(depth1 >= tree_depth):
            summary_train_model = summary_train_model.append(pd.Series([nodeOwnNum,np.nan,np.nan,np.argmax([len(dataset[dataset['class'] == 0]['class']),len(dataset[dataset['class'] == 1]['class'])]),"LN",parent,np.nan,np.nan],index= cols), ignore_index=True)
        elif(depth1 >= tree_depth):
            summary_train_model = summary_train_model.append(pd.Series([nodeOwnNum,np.nan,np.nan,np.argmax([len(dataset[dataset['class'] == 0]['class']),len(dataset[dataset['class'] == 1]['class'])]),"LN",parent,np.nan,np.nan],index= cols), ignore_index=True)
        else:
            summary_train_model = summary_train_model.append(pd.Series([nodeOwnNum,np.nan,np.nan,dataset.iloc[0,0],"LN",parent,np.nan,np.nan],index= cols), ignore_index=True)

    else:
        datasetSplitSummary = information_gain(dataset)
        datasetColName = datasetSplitSummary[0]
        datasetSplitValue = datasetSplitSummary[1]
        summary_train_model = summary_train_model.append(pd.Series([nodeOwnNum,datasetColName,datasetSplitValue,np.nan,"IN",parent,child0,child1],index= cols), ignore_index=True)

        split0 = dataset[dataset[datasetColName] == datasetSplitValue]
        split1 = dataset[dataset[datasetColName] != datasetSplitValue]

        if(split0.shape[0] != 0):
            depth1 = depth1 +1
            grow_tree(split0,child0,nodeOwnNum,sequenceGenerator(),sequenceGenerator(),depth1,depth2)

        if(split1.shape[0] != 0):
            depth2 = depth2 +1
            grow_tree(split1,child1,nodeOwnNum,sequenceGenerator(),sequenceGenerator(),depth1,depth2)
    parent = parent + 1

## Initialize the depth list
total_accuracy = []
depth = list(range(2, 17, 2))

for i in depth:
    print("---------------------------------------Depth: "+str(i)+"-------------------------------------------")
    #initialize parameters
    seqValue = 0
    cols=['NodeNum','SplitColumn','SplitValue','ClassLable','NodeType','Parent','Child0','Child1']
    summary_train_model = pd.DataFrame(np.nan, index=[0], columns=cols)
    nodeOwnNum = 0
    child0 = 1
    child1 = 2
    parent = 0
    depth1 = 0
    depth2 = 0
    tree_depth = i

    # grow the tree
    print("Growing Tree....")
    grow_tree(train,nodeOwnNum,parent,child0,child1,depth1,depth2)
    summary_train_model = summary_train_model.iloc[1:,:]
    print("Tree has been grown successfully. Now Predicting....")

    #predict the class of test record
    predictedValues = predict_value(summary_train_model,test)
    predictedValues = predictedValues.set_index([list(range(len(test)))])
    predictedValues["test_label"] = test["class"]

    acc = tree_accuracy(predictedValues)
    total_accuracy.append(acc)

print(total_accuracy)