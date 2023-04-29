from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from utils.loadData import dataset
from utils.menu import plot_options

def forestClassifier():
    png_name = 'forestClassifier'
    feature_cols = ['qualidade_da_pressao','pulso','frequencia_respiratoria','gravidade']
    target_cols = ['estado']

    ## Funciona pra 50 arvores
    n_estimators = 25

    features = dataset[feature_cols]
    target = dataset[target_cols]

    x_train, x_test, y_train, y_test = train_test_split(features, target,  test_size=0.2, random_state=1)

    clf = RandomForestClassifier(n_estimators=n_estimators, criterion='gini', max_depth=3)

    clf = clf.fit(x_train, y_train)

    predict = clf.predict(x_test)

    plot_options(clf, feature_cols, png_name, n_estimators)

