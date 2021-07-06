import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score,cross_val_predict


def cross_val(model,train_loader,optimizer, loss_func,epoch):
    lstm, classifier = model
    lstm.train()
    # lenet5.train()
    # CNN.train()
    classifier.train()

    # accuracy = cross_val_score(model,data,label,cv=10,scoring='accuracy')#for classification   精度
    y_pred = cross_val_predict(classifier(lstm),data,label,cv=k)

    # loss = -cross_val_score(model,data,label,cv=10,scoring='neg_mean_squared_error')#for regression    损失函数
    k_accuracy.append(accuracy.mean())#计算均值得分
    k_loss.append(loss.mean())
