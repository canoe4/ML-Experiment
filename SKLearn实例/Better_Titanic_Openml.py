# 使用网格搜索GridSearchCV对参数进行调优，再通过管道训练泰坦尼克数据集

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
import pandas as pd
import matplotlib.pyplot as plt

# 调优参数
param = {'columntransformer__pipeline-2__simpleimputer__strategy': ['mean', 'median'],
        'logisticregression__C': [0.1, 1.0, 10]}
grid = GridSearchCV(pipe, param_grid=param, cv=5, n_jobs=1)

grid.fit(X_train, y_train)
print(grid.score(X_test, y_test))
# 交叉验证得分
scores = cross_validate(grid, X, y, scoring='balanced_accuracy', cv=5, n_jobs=-1, return_train_score=True)
# 绘制箱型图
df_scores = pd.DataFrame(scores)
df_scores[['train_score', 'test_score']].boxplot()
plt.show()
