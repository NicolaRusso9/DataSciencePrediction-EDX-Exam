import pandas as pd
import numpy as np
from sklearn import tree
import graphviz
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('adult.csv', sep=',')
len(df)

# Remove invalid data from table
df = df[(df.astype(str) != ' ?').all(axis=1)]
len(df)

df.head()

df['income_bin'] = df.apply(lambda row: 1 if '>50K'in row['income'] else 0, axis=1)
df = df.drop(['income','fnlwgt','capital-gain','capital-loss','native-country'], axis=1)
df.head()

df.tail()

# Use one-hot encoding on categorial columns
df = pd.get_dummies(df, columns=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender'])
df.head()

# Create a sample csv for prediction
df.iloc[[0]].to_csv('predict.csv', sep=',', encoding='utf-8', index=False)

# shuffle rows
df = df.sample(frac=1)

# split training and testing data
d_train = df[:25000]
d_test = df[25000:]
d_train_att = d_train.drop(['income_bin'], axis=1)
d_train_gt50 = d_train['income_bin']
d_test_att = d_test.drop(['income_bin'], axis=1)
d_test_gt50 = d_test['income_bin']
d_att = df.drop(['income_bin'], axis=1)
d_gt50 = df['income_bin']

# number of income > 50K in whole dataset:
print("Income >50K: %d out of %d (%.2f%%)" % (np.sum(d_gt50), len(d_gt50), 100*float(np.sum(d_gt50)) / len(d_gt50)))

# Fit a decision tree
t = tree.DecisionTreeClassifier(criterion='entropy', max_depth=7)
t = t.fit(d_train_att, d_train_gt50)

# Visualize tree
data = tree.export_graphviz(t, out_file=None, label='all', impurity=False, proportion=True, feature_names=list(d_train_att), class_names=['lt50K', 'gt50K'], filled=True, rounded=True)
graphics = graphviz.Source(data)
graphics

scores = cross_val_score(t, d_att, d_gt50, cv=5)

# Show avarage score and +/- two standard deviations away (covering 95% or scores)
print('Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std()*2))

t.score(d_test_att, d_test_gt50)

sample_df = pd.read_csv('predict.csv', sep=',')
sample_df = sample_df.drop(['income_bin'], axis=1)

predict_value = sample_df.iloc[0]
y_predict = t.predict([predict_value.tolist()])
y_predict[0]

for max_depth in range(1, 20):
    t = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
    scores = cross_val_score(t, d_att, d_gt50, cv=5)
    print("Max depth: %d <==> Accuracy: %0.2f (+/- %0.2f)" % (max_depth, scores.mean(), scores.std()*2))