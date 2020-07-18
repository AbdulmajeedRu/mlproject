from sklearn.neighbors import NearestNeighbors
import sklearn
import numpy as np

#Open files
NegRev = open('books/negative.review', 'r')
PosRev=open('books/positive.review', 'r')
unlabeledR = open('books/unlabeled.review', 'r')

#Read the files and split them by lines (each line represents one review)
Train_pos = PosRev.read().splitlines()
Train_neg = NegRev.read().splitlines()
Test_reviews = unlabeledR.read().splitlines()

# Close the file pointers
NegRev.close()
PosRev.close()
unlabeledR.close()

# Combine the training sets to deal with one training set

trainset = [(x,1) for x in Train_pos] + [(x, -1) for x in Train_neg]
print ("The total number of training data is:", len(trainset))


# Extract the features from the training data
features_dic={}
id_=0
for review, label in trainset:
    features = review.strip().split()[:-1]
    for item in features:
        feat,val = item.strip().split(":")
        if feat not in features_dic:
            features_dic[feat] = id_
            id_+=1
print ("The total number of features is", len(features_dic))


#represent the reviews using the obtained features
X = np.zeros((len(trainset), len(features_dic)))
y = [y for (x,y) in trainset]

# We need to fill the matrix proberly now using feature ids
for i, (review, label) in enumerate(trainset):