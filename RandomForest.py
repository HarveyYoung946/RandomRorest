from sklearn.model_selection import  train_test_split,cross_val_score,GridSearchCV
from sklearn.datasets import make_blobs
from sklearn.ensemble import  RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import sklearn.metrics as metrics

#共创建100个类共10000个样本，每个样本10个特征(指标)
x,y = make_blobs(n_samples=10000,n_features=10,centers=100,random_state=0)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=6)
#print(y_test[3])

##决策树
cf1 = DecisionTreeClassifier(max_depth=None,min_samples_split=2,random_state=0)
cf1.fit(x_train,y_train)
scores1 = cross_val_score(cf1,x_test,y_test)
print(scores1.mean())

##随机森林,n_estimators为迭代次数，即决策树的个数
cf2 = RandomForestClassifier(n_estimators=10,max_depth=None,min_samples_split=2,random_state=1,max_features='sqrt')
cf2.fit(x_train,y_train)
#获取各个变量的重要性
#importance = cf2.feature_importances_
#print(importance)
scores2 = cross_val_score(cf2,x_test,y_test)
print(scores2.mean())

##ExtraTree 分类器集合，与随机森林的不同点：每棵决策树选择划分点的方式不同。对于普通决策树，每个特征都是根据某个标准
# (信息增益或者gini不纯)去进行划分，比如说一个数值特征的取值范围是0到100，当我们把这个特征划分为0-45，45-100的时候，
# 效果最佳（从信息增益或者gini不纯的角度），决策树就会将45作为划分点；然后再按照某个标准来选择特征。而对于extra trees
# 中的决策树，划分点的选择更为随机，比如一个数值特征的取值范围是0到100，我们随机选择一个0到100的数值作为划分点；然后再
# 按照评判标准选择一个特征。
cf3 = ExtraTreesClassifier(n_estimators=10,max_depth=None,min_samples_split=2,random_state=1)
cf3.fit(x_train,y_train)
scores3 = cross_val_score(cf3,x_test,y_test)
print(scores3.mean())

'''
#使用网格搜索参数
param_grid = { 'criterion':['entropy','gini'],
               'max_depth':[10,50,100],
               'min_samples_split':[2,4,8,12,16,20,24,28]
}
clf = DecisionTreeClassifier()
#传入参数,param_grid网格搜索参数,scoring='roc_auc'评估指标,cv交叉验证次数
clfcv = GridSearchCV(estimator=clf,param_grid=param_grid,cv=4)
#模型训练
clfcv.fit(x_train,y_train)
test_result = clfcv.predict(x_test)

#模型评估

print('决策树准确度')
print(metrics.classification_report(y_test,test_result))

#决策树的AUC,对于scoring='roc_auc'是用来检测定性数据结果的，比如好人坏人，是和否等，即结果为0或1的数据预测(二分类)。
#print('决策树的AUC')
#fpr_test,tpr_test,th_test = metrics.roc_curve(y_test,test_result)
#print('AUC=%.4f'%metrics.auc(fpr_test,tpr_test))

#网络搜索最优参数
print(clfcv.best_params_)
'''
last_clf = DecisionTreeClassifier(criterion='entropy',max_depth=10,min_samples_split=2)
last_clf.fit(x_train,y_train)
scores3 = cross_val_score(last_clf,x_test,y_test)
print(scores3.mean())
#变量相应重要性
#print(last_clf.feature_importances_)
#last_clf.predict(x_test)
#绘制图形，graphviz安装教程见https://www.pythonf.cn/read/129001
import graphviz
dot_data = tree.export_graphviz(last_clf,out_file=None,filled=True,rounded=True)
graph = graphviz.Source(dot_data)
graph.render("user")
#from PIL import Image
#Image.open('user.jpg')
