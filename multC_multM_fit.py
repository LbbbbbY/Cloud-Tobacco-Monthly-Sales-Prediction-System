import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import matplotlib.pyplot as plt 
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签 
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

file_path = r'./数据/地市-月/'
file_name = '软大重九'
pcom_id = '所有地市'
version = 'V3'
allDf = pd.read_excel(file_path+"{}-{}-xgboost销量预测结果-{}-201811-202008.xlsx".format(
        file_name,pcom_id,version),sheet_name='Sheet1')
#allDf = pd.read_excel(file_path+"{}-{}-moveAvg3M销量预测结果.xlsx".format(file_name,pcom_id))
allDf = allDf.drop('%',axis=1)
allDf.columns = ['com_id','month','x1_val','y_val']
allDf['month'] = allDf['month'].map(lambda x: str(x))
monthList = sorted(set(allDf["month"]))
allDf['x2_val'] = allDf['y_val'].map(lambda x: round(x*random.uniform(0.9,1.1),4))

"标准误差"
def stdError_func(y_test, y):
  return np.sqrt(np.mean((y_test - y) ** 2))

"对比该模型预测与均值预测跟实际值的差距平方和"
def R2_1_func(y_test, y):
  return 1 - ((y_test - y) ** 2).sum() / ((y.mean() - y) ** 2).sum()

"对比该模型预测与均值预测跟实际值的标准误差"
def R2_2_func(y_test, y):
  y_mean = np.array(y)
  y_mean[:] = y.mean()
  return 1 - stdError_func(y_test, y) / stdError_func(y_mean, y)

"拟合的月份"
beginMonth = '201912'
beginMIndex = monthList.index(beginMonth)
endMonth = '202008'
endMIndex = monthList.index(endMonth)

outputList = []
comList = sorted(set(allDf["com_id"]))
for curCom in comList:
    print('【curCom is {}】'.format(curCom))
    df = allDf[allDf['com_id']==curCom]
    for tgtM in monthList[beginMIndex:endMIndex+1]:
        print('--【tgtM is {}】'.format(tgtM))
        targetMIndex = monthList.index(tgtM)
        trainLen = 12
        trainML = monthList[targetMIndex-trainLen:targetMIndex]
        "训练集输入与输出"
        trainDf = df[df['month'].isin(trainML)].copy()
        if trainDf.shape[0] == 0:
            print('----【trainDf.shape[0] == 0 skipped】')
            continue
        train_x = trainDf[['x1_val','x2_val']].values
        train_y = trainDf['y_val'].values
        "特征处理"
        poly_reg = PolynomialFeatures(degree=1)
        train_X = poly_reg.fit_transform(train_x)
        "拟合"
        lin_reg_2 = linear_model.LinearRegression()
        lin_reg_2.fit(train_X,train_y)
        "预测targetM月的销量"
        preDf = df[df['month'].isin([tgtM])].copy()
        if preDf.shape[0] == 0:
            print('----【preDf.shape[0] == 0 skipped】')
            continue
        pre_x = preDf[['x1_val','x2_val']].values
        pre_X = poly_reg.fit_transform(pre_x)
        pre_y = lin_reg_2.predict(pre_X)
        "当填报的政策值为0时，拟合值等于0；当拟合值小于零时，等于填报的政策值"
        if pre_x[0][1] == 0:
            pre_y = np.array([0])
        else:
            if pre_y[0] <= 0:
                pre_y = np.array([pre_x[0][1]])
        preDf['fit_val'] = pre_y
        preDf['%'] = round(1-abs(preDf['fit_val']-preDf['y_val'])/preDf['y_val'], 4)*100
        outputList.append(preDf)
        
outputDf = pd.concat(outputList)
columns = outputDf.columns.tolist()
nameDf = pd.read_csv(file_path+"全国地市名.csv", sep=',')
outputDf = outputDf.merge(nameDf,on="com_id",how="left")
outputDf = outputDf[['pcom_name','pcom_id','com_name']+columns]
print("【精度】：\n{}".format(outputDf))
outputDf['%'] = outputDf['%'].map(lambda x: 0 if np.isinf(x) else x)
outputDf.to_excel(file_path+"{}-{}-fit结果-{}-{}.xlsx".format(
        file_name,pcom_id,beginMonth,endMonth), index=False)
plt.hist(outputDf['%'], bins=50)

