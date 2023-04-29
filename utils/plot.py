from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def plot(clf, feature_cols, png_name, n_estimators=1):
    fig, axes = plt.subplots(nrows = 1, ncols=n_estimators, figsize=(n_estimators*1.8,2), dpi=600)
    if n_estimators != 1:
        for i in range(n_estimators):
            plot_tree(clf.estimators_[i], feature_names=feature_cols, rounded=True, ax=axes[i])
    else:
        for i in range(n_estimators):
            plot_tree(clf, feature_names=feature_cols, rounded=True, ax=None)
            
    
            
    fig.savefig(f'{png_name}.png')
    
def importance(clf):
    # Creating importances_df dataframe
    importances_df = pd.DataFrame({"feature_names" : clf.feature_names_in_, 
                                "importances" : clf.feature_importances_})
                                
    # Plotting bar chart, g is from graph
    g = sns.barplot(x=importances_df["feature_names"], 
                    y=importances_df["importances"])
    g.set_title("Feature importances", fontsize=14) 
    plt.show()

