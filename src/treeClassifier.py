from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from utils.loadData import dataset
from utils.menu import plot_options

def treeClassifier():
    png_name = 'treeClassifier'
    feature_cols = ['qualidade_da_pressao','pulso','frequencia_respiratoria','gravidade']
    target_cols = ['estado']

    features = dataset[feature_cols]
    target = dataset[target_cols]

    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    clf = DecisionTreeClassifier(criterion='gini', max_depth=3)

    clf = clf.fit(x_train, y_train)

    predict = clf.predict(x_test)
    print("predict", predict)

    accuracy = accuracy_score(y_test, predict)
    print("accuracy:", accuracy)
    
    precision = precision_score(y_test, predict, average=None)
    print("precision:", precision)
    
    recall = recall_score(y_test, predict, average=None)
    print("recall:", recall)
    
    f1 = f1_score(y_test, predict, average=None)
    print("f1 score:", f1)

    plot_options(clf, feature_cols, png_name, x_test, y_test, type='tree', classifier='true')

