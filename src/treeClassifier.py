from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from utils.loadData import dataset
from utils.plot import plot, importance

png_name = 'treeClassifier'
feature_cols = ['qualidade_da_pressao','pulso','frequencia_respiratoria','gravidade']
target_cols = ['estado']

features = dataset[feature_cols]
target = dataset[target_cols]

x_train, x_test, y_train, y_test = train_test_split(features, target,  test_size=0.2, random_state=1)

clf = DecisionTreeClassifier(criterion='gini', max_depth=3)

clf = clf.fit(x_train, y_train)

predict = clf.predict(x_test)

accuracy = accuracy_score(y_test, predict)

print(accuracy)

importance(clf)
plot(clf,feature_cols, png_name)
