import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import classification_report
from sklearn import tree
from graphviz import Source
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('adult.data.gz', compression='gzip')
df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
              'relationship','race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
              'native-country', 'salary']

print(df.info())
print(df.head())

X = df.drop(['salary'], axis=1)
y = np.where(df['salary'] == ' <=50K', 0, 1)

df['capital-gain'] = df['capital-gain'].astype(float)
print(X.select_dtypes(include='object').tail(20))

categorical_columns = [c for c in X.columns if X[c].dtype.name == 'object']
for c in categorical_columns:
    X[c] = np.where(X[c] == ' ?', X[c].mode(), df[c])
print(X.select_dtypes(include='object').tail(20))

X = pd.concat([X, pd.get_dummies(X.select_dtypes(include='object'))], axis=1)
X = X.drop(['workclass', 'education', 'marital-status', 'occupation',
       'relationship', 'race', 'sex', 'native-country'], axis=1)
print(X.head())

# Dividing the data into training and test sub-data groups, building a decision tree model and ‘fitting’ the model to the training data.
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)
d_tree1 = DecisionTreeClassifier(max_depth = 2, random_state=42)
d_tree1.fit(X_train, y_train)

# We give the model the test data that we do not see and make a prediction.
predictions = d_tree1.predict(X_test)
score = round(accuracy_score(y_test, predictions), 3)
cm1 = cm(y_test, predictions)
sns.heatmap(cm1, annot=True, fmt=".0f")
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Accuracy Score: {0}'.format(score), size = 15)

# Model success metrics: Precision, recall, f1-score
print(classification_report(y_test, predictions, target_names=['<=50K', '>50K']))

# Visualizing the decision tree
dot_data = tree.export_graphviz(d_tree1, out_file=None, feature_names=X.columns, filled=True)
graph = Source(dot_data)
graph.view()

# Analyze the importance ranking of the attributes of the model.
plt.figure(figsize=(16, 9))
d_tree2 = DecisionTreeClassifier(max_depth=8, random_state=42)
d_tree2.fit(X_train, y_train)
ranking = d_tree2.feature_importances_
features = np.argsort(ranking)[::-1][:10]
columns = X.columns
plt.title("Feature importances based on Decision Tree Classifier", y = 1.03, size = 18)
plt.bar(range(len(features)), ranking[features], color="lime", align="center")
plt.xticks(range(len(features)), columns[features], rotation=80)
plt.show()