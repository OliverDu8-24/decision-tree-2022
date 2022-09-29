import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split

from pipeline import pipeline

def train(feature, label, feature_names, test_size=0.2, id=0, cls='entropy'):
    x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=test_size, shuffle=True)

    clf = tree.DecisionTreeClassifier(criterion=cls)
    clf.fit(x_train, y_train)

    s = clf.score(x_test, y_test)
    print(f"accuracy: {s}")

    with open(f"./results/out-{id}.dot", 'w') as f :
        f = tree.export_graphviz(clf, out_file=f,
                feature_names=feature_names)

    dotdata = tree.export_graphviz(clf, out_file=f,
                feature_names=feature_names)

    import os  
    os.environ['PATH'] = os.pathsep + r'D:\tools\code\Graphviz\bin'
    os.system(f'dot -Tpng ./results/out-{id}.dot -o ./results/决策树模型-{id}-accuracy-{s:.3f}.png')

epochs = 10
feature, label, feature_name = pipeline()
if __name__ == "__main__":
    for epoch in range(epochs):
        # train(feature, label, feature_name, cls='gini')
        train(feature, label, feature_name, id=epoch, cls='entropy')