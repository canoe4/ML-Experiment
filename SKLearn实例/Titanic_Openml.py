# 用同样的方法训练乳腺癌分类器

# 导入数据集并对数据集进行独立分割
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)
print(len(X))
print(y[0])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=12)
print(X_train)

# 使用GradientBoostingClassifier梯度提升分类器建模进行训练和测试
from sklearn.ensemble import GradientBoostingClassifier
clf1 = GradientBoostingClassifier(n_estimators=100, random_state=12)
clf1.fit(X_train, y_train)
accuracy11 = clf1.score(X_test, y_test)

# 这里提供另一种测试精度的方法，使用predict预测函数对X_test进行预测得到y_pred，再调用balanced_accuracy_score评估方法对y_pred和y_test进行比对，进行精准度测试
from sklearn.metrics import balanced_accuracy_score
y_pred = clf1.predict(X_test)
accuracy12 = balanced_accuracy_score(y_test, y_pred)
print(accuracy11)
print(accuracy12)


# 分别使用两种模型进行训练
from sklearn.linear_model import LogisticRegression
clf2 = LogisticRegression(solver='lbfgs', max_iter=5000,
                          multi_class='ovr', random_state=12)
clf2.fit(X_train, y_train)
accuracy2 = clf2.score(X_test, y_test)
print(accuracy2)

from sklearn.ensemble import RandomForestClassifier
clf3 = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=12)
clf3.fit(X_train, y_train)
accuracy3 = clf3.score(X_test, y_test)
print(accuracy3)
