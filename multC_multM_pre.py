# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 09:23:48 2020
多地市多月预测
@author: jiangzhr
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import time
import dataDeal
import itertools
import matplotlib.pyplot as plt 

file_name = '软大重九'
file_path = r'./数据/地市-月/'
#allDf = pd.read_csv(file_path+"{}.csv".format(file_name))
allDf = pd.read_excel(file_path+"{}.xlsx".format(file_name))
allDf.columns = ['pcom_id','com_id','month','qty_ord_item','need_ord_item']

"要预测的省份/地市（由于计算复杂，单省份或单地市预测）"
pcom_id = '所有地市'
#allDf = allDf[allDf['pcom_id']==pcom_id]
allDf = allDf.drop('pcom_id', axis=1)
#com_id = 11530101
#allDf = allDf[allDf['com_id'].isin([com_id])]

"将1、2月合并"
#allDf['month'] = allDf['month'].map(lambda x: str(x))
#allDf['month'] = allDf['month'].map(lambda x: x[:4]+'01' if x[4:]=='02' else x)
#allDf = allDf.groupby(['com_id','month']).agg({
#        'qty_ord_item':'sum',
#        'need_ord_item':'sum'})
#allDf = allDf.reset_index(drop=False)

allDf['month'] = allDf['month'].map(lambda x: str(x))
month_list = sorted(set(allDf["month"]))

"计算订单满足率"
allDf = dataDeal.cal_ddmzl_func(allDf)

"计算月占比、月累占比"
allDf = dataDeal.cal_month_ratio_func(allDf)
allDf = allDf.groupby(['com_id','year']).apply(dataDeal.cal_month_progress_func)
allDf = allDf.reset_index(drop=True)

def keep_posi(x): return max(0, x)

def my_log_1pX(x,one=1): return x

def get_neib_months(x, month_list=month_list, relative='this'):
    if relative.lower() in ['this', 'huan']:
        if x not in month_list: return ['0']*5
        idx = month_list.index(x)
        if idx<5: return ['0']*5
        return month_list[idx-5:idx-0]
    elif relative.lower() in ['prev', 'tong']:
        if x not in month_list: return ['0']*5
        idx = month_list.index(x)
        if idx<12+2: return ['0']*5
        begin_index = (idx-12-2) if (idx-12-2)>0 else 0
        end_index = idx-12+3
        months = month_list[begin_index:end_index]
        if len(months)<5: return ['0']*5
        months.sort()
        return months[-5:]
    else:
        return ['0']*5
    
"选取的特征、预测目标、忽略的月份"
cols1 = ['_qty_ord_item','_ddmzl_item']
yCol = 'qty_ord_item'
speMonth = ["01","02"]
"预测的月份"
beginMonth = '201811'
endMonth = '201911'

outputList = []
com_list = sorted(set(allDf["com_id"]))
for curCom in com_list:
    "对于新品，每个地市的上市时间不一样，month_list也不一样"
    curComDf = allDf[allDf['com_id']==curCom].copy()
    "计算同期增幅"
    curComDf = dataDeal.cal_same_month_grow_func(curComDf)
    "计算去年年累增幅"
    curComDf = dataDeal.cal_last_year_grow_func(curComDf)
    month_list = sorted(set(curComDf["month"]))
    #有的地市开始与结束月的index可能取不到
    if (beginMonth not in month_list) or (endMonth not in month_list): continue
    beginMIndex = month_list.index(beginMonth)
    endMIndex = month_list.index(endMonth)
    for tgtM in month_list[beginMIndex:endMIndex+1]:
        df = curComDf[curComDf['month'].map(lambda x: x<=tgtM)]
        print('【curCom is {},tgtM is {}】'.format(curCom,tgtM))
        tgtMIndex = month_list.index(tgtM)
        trainLen = 12*3
        train_ML = month_list[tgtMIndex-trainLen:tgtMIndex]
        # 有些地方要预测的月份没有记录，不做预测，例如佛山 11440601 202008
        if df[df['month']==tgtM].shape[0] == 0:
            print('--{}-{} is null skipped!'.format(curCom,tgtM))
            continue
        df = df.set_index(["com_id","month"])
        df = df.unstack()
        [x,y] = df.columns.levels
        df.columns = [ item[1]+"_"+item[0] for item in itertools.product(list(x),list(y)) ]
        df = df.reset_index(drop=False)
        "记录预测期真实值"
        realVal = df[['com_id', tgtM+'_'+yCol]]
        realVal.columns = ['com_id','real_val']
        
        ddList = []
        for curM in train_ML+[tgtM]:
            dd = pd.DataFrame()
            #当预测正常月份时，训练集剔除特殊月份
#            if (tgtM[4:] not in speMonth) and (curM[4:] in speMonth): 
#                print('-tgtM={0} not in {2},curM={1} in {2} skipped!'.format(tgtM, curM, speMonth))
#                continue
            #当预测特殊月份的时候，去掉其他月份
            if (tgtM[4:] in speMonth) and (curM[4:] not in speMonth): 
                print('-tgtM={0} in {2},curM!={1} in {2} skipped!'.format(tgtM, curM, speMonth))
                continue
            for cc in cols1:
                (h5, h4, h3, h2, h1) = get_neib_months(curM, month_list)
                if h5=='0': continue
                d_h5 = np.array([*map(keep_posi, df[h5 +cc])])
                d_h4 = np.array([*map(keep_posi, df[h4 +cc])])
                d_h3 = np.array([*map(keep_posi, df[h3 +cc])])
                d_h2 = np.array([*map(keep_posi, df[h2 +cc])])
                d_h1 = np.array([*map(keep_posi, df[h1 +cc])])
                
                #当预测月份不是特殊月时，加入环比的移动平均值特征
                if tgtM[4:] not in speMonth:
                    dd.loc[:,  'H1'+cc] = my_log_1pX(d_h1 , one=0.1)
                    dd.loc[:,  'H2'+cc] = my_log_1pX(d_h2 , one=0.1)
                    dd.loc[:,  'H3'+cc] = my_log_1pX(d_h3 , one=0.1)
                    dd.loc[:,  'H4'+cc] = my_log_1pX(d_h4 , one=0.1)
                    dd.loc[:,  'H5'+cc] = my_log_1pX(d_h5 , one=0.1)
                    #移动两个月
                    dd.loc[:, 'avg12'+cc] = my_log_1pX((d_h1 + d_h2)/2, one=0.1)
                    dd.loc[:, 'avg23'+cc] = my_log_1pX((d_h2 + d_h3)/2, one=0.1)
                    #移动三个月
                    dd.loc[:, 'avg123'+cc] = my_log_1pX((d_h1 + d_h2 + d_h3)/3, one=0.1)
                    dd.loc[:, 'avg234'+cc] = my_log_1pX((d_h2 + d_h3 + d_h4)/3, one=0.1)
                    #移动四个月
                    dd.loc[:, 'avg1234'+cc] = my_log_1pX((d_h1 + d_h2 + d_h3 + d_h4)/4, one=0.1)
                    dd.loc[:, 'avg2345'+cc] = my_log_1pX((d_h2 + d_h3 + d_h4 + d_h5)/4, one=0.1)
                    #移动五个月
                    dd.loc[:, 'avg12345'+cc] = my_log_1pX((d_h1 + d_h2 + d_h3 + d_h4 + d_h5)/5, one=0.1)
                    #斜率
                    dd.loc[:, 'r123'+cc] = np.polyfit([1,2,3],[d_h3,d_h2,d_h1],1)[0]
                    #标准差
                    d2t = df[[h5+cc, h4+cc, h3+cc, h2+cc, h1+cc]]
                    dd.loc[:,'S1'+cc+'_std'] = d2t.std(axis=1)
                
                "对于新品，同比时肯定取不到的，所以把同比去掉"
                (t5, t4, t3, t2, t1) = get_neib_months(curM, month_list, relative='tong')
                if t5=='0': continue
                d_t5 = np.array([*map(keep_posi, df[t5 +cc])])
                d_t4 = np.array([*map(keep_posi, df[t4 +cc])])
                d_t3 = np.array([*map(keep_posi, df[t3 +cc])])
                d_t2 = np.array([*map(keep_posi, df[t2 +cc])])
                d_t1 = np.array([*map(keep_posi, df[t1 +cc])])
                
                if tgtM[4:] in speMonth:
                    dd.loc[:,  'T3'+cc] = my_log_1pX(d_t3 , one=0.1)
                else:
                    dd.loc[:,  'T5'+cc] = my_log_1pX(d_t5 , one=0.1)
                    dd.loc[:,  'T4'+cc] = my_log_1pX(d_t4 , one=0.1)
                    dd.loc[:,  'T3'+cc] = my_log_1pX(d_t3 , one=0.1)
                    dd.loc[:,  'T2'+cc] = my_log_1pX(d_t2 , one=0.1)
                    dd.loc[:,  'T1'+cc] = my_log_1pX(d_t1 , one=0.1)
                    #移动两个月
                    dd.loc[:, 'Tavg23'+cc] = my_log_1pX((d_t2 + d_t3)/2, one=0.1)
                    dd.loc[:, 'Tavg34'+cc] = my_log_1pX((d_t3 + d_t4)/2, one=0.1)
                    #移动三个月
                    dd.loc[:, 'Tavg123'+cc] = my_log_1pX((d_t1 + d_t2 + d_t3)/3, one=0.1)
                    dd.loc[:, 'Tavg234'+cc] = my_log_1pX((d_t2 + d_t3 + d_t4)/3, one=0.1)
                    dd.loc[:, 'Tavg345'+cc] = my_log_1pX((d_t3 + d_t4 + d_t5)/3, one=0.1)
                    #移动四个月
                    dd.loc[:, 'Tavg1234'+cc] = my_log_1pX((d_t1 + d_t2 + d_t3 + d_t4)/4, one=0.1)
                    dd.loc[:, 'Tavg2345'+cc] = my_log_1pX((d_t2 + d_t3 + d_t4 + d_t5)/4, one=0.1)
                    #移动五个月
                    dd.loc[:, 'Tavg12345'+cc] = my_log_1pX((d_t1 + d_t2 + d_t3 + d_t4 + d_t5)/5, one=0.1)
                    #斜率
                    dd.loc[:, 'Tr543'+cc] = np.polyfit([1,2,3],[d_t5,d_t4,d_t3],1)[0]
                    #标准差
                    d2t = df[[t5+cc, t4+cc, t3+cc, t2+cc, t1+cc]]
                    dd.loc[:,'tS1'+cc+'_std'] = d2t.std(axis=1)
                    #本期/同期
                    if d_t4 == 0:
                        dd.loc[:, 'H1/T4'+cc] = 1
                    else:
                        dd.loc[:, 'H1/T4'+cc] = my_log_1pX((d_h1 / d_t4), one=0.1)
                    if d_t5 == 0:
                        dd.loc[:, 'H2/T5'+cc] = 1
                    else:
                        dd.loc[:, 'H2/T5'+cc] = my_log_1pX((d_h2 / d_t5), one=0.1)
                    
                "同期的销量占比、及累计销量占比"
                if cc == "_qty_ord_item":
                    r_t3 = np.array([*map(keep_posi, df[t3 +'_ratio'])])
                    dd.loc[:, 'T'+'_ratio'] = my_log_1pX(r_t3 , one=0.1)
                    ra_t3 = np.array([*map(keep_posi, df[t3 +'_ratio_add'])])
                    dd.loc[:, 'T'+'_ratio_add'] = my_log_1pX(ra_t3 , one=0.1)
                    #在预测特殊月份的时候加入去年年销量增幅、同期增幅
                    if tgtM[4:] in speMonth:
                        lastYearGrow = np.array([*map(keep_posi, df[t3 +'_last_year_grow'])])
                        dd.loc[:, 'T3'+'_last_year_grow'] = my_log_1pX(lastYearGrow , one=0.1)
                        sameMonthGrow = np.array([*map(keep_posi, df[t3 +'_same_month_grow'])])
                        dd.loc[:, 'T3'+'_same_month_grow'] = my_log_1pX(sameMonthGrow , one=0.1)
            
            if h5=='0' or t5=='0': # 当同比与环比取不到时，跳过
                print(curM, 'skipped!')
                continue
            
            if curM==tgtM:
                df[curM+'_'+yCol]=0 # 将要预测周期该指标的值置0
                
            cur_y = np.array(df[curM+'_'+yCol])
            dd.loc[:, yCol] = cur_y
            dd.fillna(0, inplace=True)
            ddList.append(dd)
        
        "当可用训练集数据少于4个月时，跳过"
        if len(ddList) < 4:
            print('--len(input) < 4 skipped')
            continue
        
        dnx_lst=[]
        dny_lst=[]
        for dTmp in ddList[:-1]: # 循环前 n-1 个序列，将训练集的输入输出序列化
            dny_lst.append(dTmp[yCol])
            dTmp.pop(yCol)
            dnx_lst.append(dTmp)
        
        dnx_l=pd.concat(dnx_lst)
        dny_l=pd.concat(dny_lst)
        
        rnd_num = round(time.time()*1000)%1000000
        my_clf = xgb.XGBRegressor(n_estimators=60,
                                  learning_rate=0.1,
                                  min_samples_leaf=2,
                                  max_depth=5,
                                  random_state=rnd_num)
        my_clf.fit(dnx_l, dny_l)
        
        # 取预测期的输入
        dTmp=ddList[-1]
        dTmp.pop(yCol)
        
        # 预测期输出
        dny_p = my_clf.predict(dTmp)
        
        df[tgtM+"_"+yCol]=dny_p
        preVal = df[['com_id', tgtM+"_"+yCol]]
        preVal.columns = ['com_id','pre_val']
        resVal=preVal.merge(realVal,on="com_id",how="inner")
        resVal['%'] = round(1-abs(resVal['pre_val']-resVal['real_val'])/resVal['real_val'], 4)*100
        resVal.insert(1,'month',tgtM)
#        print("--【精度】：\n{}".format(resVal))
        outputList.append(resVal)
        
outputDf = pd.concat(outputList)
print("【精度】：\n{}".format(outputDf))
outputDf['%'] = outputDf['%'].map(lambda x: 0 if np.isinf(x) else x)
#columns = outputDf.columns.tolist()
#nameDf = pd.read_csv(file_path+"全国地市名.csv", sep=',')
#outputDf = outputDf.merge(nameDf,on="com_id",how="left")
#outputDf = outputDf[['pcom_name','pcom_id','com_name']+columns]
outputDf.to_excel(file_path+"{}-{}-xgboost销量预测结果-V3-{}-{}.xlsx".format(
        file_name,pcom_id,beginMonth,endMonth), index=False)
plt.hist(outputDf['%'], bins=50)

