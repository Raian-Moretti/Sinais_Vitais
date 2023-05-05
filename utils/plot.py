from sklearn.tree import plot_tree
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot(clf, feature_cols, png_name, n_estimators):
    plt.clf()
    fig, axes = plt.subplots(nrows = 1, ncols=n_estimators, figsize=(n_estimators*1.8,2), dpi=600)
    if n_estimators != 1:
        for i in range(n_estimators):
            plot_tree(clf.estimators_[i], feature_names=feature_cols, rounded=True, ax=axes[i])
    else:
        for i in range(n_estimators):
            plot_tree(clf, feature_names=feature_cols, rounded=True, ax=None)
            
    fig.savefig(f'images/plot_{png_name}.png')
    plt.close()
    
def importance(clf,png_name):
    plt.clf()
    # Creating importances_df dataframe
    importances_df = pd.DataFrame({"feature_names" : clf.feature_names_in_, 
                                "importances" : clf.feature_importances_})
                                
    # Plotting bar chart, g is from graph
    g = sns.barplot(x=importances_df["feature_names"], 
                    y=importances_df["importances"])
    g.set_title("Feature importances", fontsize=14) 
    plt.savefig(f'images/importance_{png_name}.png')
    plt.show()
    plt.close()

def confusion_matrix(clf, x_test, y_test,png_name):
    fig=ConfusionMatrixDisplay.from_estimator(clf, x_test, y_test,display_labels=["1","2","3","4"])
    fig.figure_.suptitle("Confusion Matrix")
    plt.savefig(f'images/confusion_matrix_{png_name}.png')
    plt.show()
    plt.close()