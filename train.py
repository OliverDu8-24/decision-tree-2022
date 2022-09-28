import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split

from pipeline import pipeline

def train(feature, label, feature_names, cls='entropy'):
    x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.33, shuffle=True)

    clf = tree.DecisionTreeClassifier(criterion=cls)
    clf.fit(x_train, y_train)

    s = clf.score(x_test, y_test)
    print(f"accuracy: {s}")

    with open("out.dot", 'w') as f :
        f = tree.export_graphviz(clf, out_file=f,
                feature_names=feature_names)

    dotdata = tree.export_graphviz(clf, out_file=f,
                feature_names=feature_names)

    import os  
    os.environ['PATH'] = os.pathsep + r'D:\tools\code\Graphviz\bin'
    os.system('dot -Tpng out.dot -o 决策树模型.png')

epochs = 10
feature, label, feature_name = pipeline()
if __name__ == "__main__":
    for i in range(epochs):
        # train(feature, label, feature_name, cls='gini')
        train(feature, label, feature_name, cls='entropy')