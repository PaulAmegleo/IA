import pandas
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


df = pandas.read_csv("Data.csv")

d = {'Up': 0, 'Down': 1, 'Left': 2, 'Right': 3}
df['Movement'] = df['Movement'].map(d)

features = ['x0','x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17','x18','x19','x20',
            'y0','y1','y2','y3','y4','y5','y6','y7','y8','y9','y10','y11','y12','y13','y14','y15','y16','y17','y18','y19','y20',
            'z0','z1','z2','z3','z4','z5','z6','z7','z8','z9','z10','z11','z12','z13','z14','z15','z16','z17','z18','z19','z20']

X = df[features]
y = df['Movement']

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X,y)


class_names = ['Up', 'Down', 'Left', 'Right']
tree.plot_tree(dtree, class_names=class_names)
plt.show()