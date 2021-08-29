import pandas as pd
import sklearn
from sklearn import svm, preprocessing

FILE = 'diamonds.csv'
df = pd.read_csv(FILE, index_col=0)
df.head()


# to transform into numbers with regards to meaning and sequence
cut_dict = {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5}
color_dict = {'J': 1, 'I': 2, 'H': 3, 'G': 4, 'F': 5, 'E': 6, 'D': 7}
clarity_dict = {'I1': 1 , 'SI2': 2, 'SI1': 3, 'VS2': 4, 'VS1': 5, 'VVS2': 6, 'VVS1': 7, 'IF': 8}

# Transforming categorical data into numerical data
#using apply lambda
df['cut'] = df['cut'].apply(lambda c: cut_dict[c])
#using map
df['color'] = df['color'].map(color_dict)
df['clarity'] = df['clarity'].map(clarity_dict)

#shuffle dataframe to avoid bias in model
df = sklearn.utils.shuffle(df)

# X & y for the model
# X: all the relevent data required for the predictuion
# y: the values to be predicted by the model
X = df.drop("price", axis=1).values
# Scale X to produce more uniform values
X = preprocessing.scale(X)
y = df["price"].values

# define test sample, to test the model after it finishes. the sample will never be fed to training the model
test_sample = 100 
#since data is already shuffled, taking last 200 is OK as a random sample
X_train = X[:-test_sample]
y_train = y[:-test_sample]

X_test = X[-test_sample:]
y_test = y[-test_sample:]

# train the the model for prediction
clf = svm.SVR(kernel = 'linear')
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

# test the model with the test sample
for x, y in zip(X_test, y_test):
    prediction = clf.predict([x])[0]
    if prediction < 0:
        print('!!! Invalid Value')
    print('Model: {}, Actual: {}'.format(prediction, y))





