import numpy as np 
import pandas as pd 
import os
from scipy.linalg import norm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer



def tf_similarity(sentence1, sentence2):
    def add_space(s):
        return ' '.join(list(s))
    # 将字中间加入空格
    sentence1, sentence2 = add_space(sentence1), add_space(sentence2)
    # 转化为TF矩阵
    cv = CountVectorizer(tokenizer=lambda s: s.split())
    corpus = [sentence1, sentence2]
    vectors = cv.fit_transform(corpus).toarray()
    # 计算TF系数
    return np.dot(vectors[0], vectors[1]) / (norm(vectors[0]) * norm(vectors[1]))
 
def tfidf_similarity(sentence1, sentence2):
    def add_space(s):
        return ' '.join(str(list(s)))
    # 将字中间加入空格
    sentence1, sentence2 = add_space(sentence1), add_space(sentence2)
    # 转化为TF矩阵
    cv = TfidfVectorizer(tokenizer=lambda s: s.split())
    corpus = [sentence1, sentence2]
    vectors = cv.fit_transform(corpus).toarray()
    # print(vectors)
    # 计算TFIDF系数
    return np.dot(vectors[0], vectors[1]) / (norm(vectors[0]) * norm(vectors[1]))

#Calculate similarity and deduction points
def calaulate_similarity_and_score(smart_guard_data,filterd_data):
    # global smart_guard_data
    for i in range(len(filterd_data.index)):
        idex1 = filterd_data.index[i]
        s1 = smart_guard_data.loc[idex1,'corrective_action']
        s3 = smart_guard_data.loc[idex1,'preventive_action']
        t1 = smart_guard_data.loc[idex1,'event_date']

        if s1 == 'nan':
            smart_guard_data.loc[idex1,'ca_score'] = -100
        else:
            for j in range( i+1 , len(filterd_data.index) ):
                # print('%d=>%d'%(j,filterd_data.index[j]))
                idex2 = filterd_data.index[j]
                s2 = smart_guard_data.loc[idex2,'corrective_action']
                t2 = smart_guard_data.loc[idex2,'event_date']
                sc = tf_similarity(s1, s2)   #計算相似分數
                #如果時間差在2年內
                if abs(t1 -t2) <pd.Timedelta(days = 730):       
                    ##############################且如果相似度高於 80%##############################
                    if(sc >0.8):
                        smart_guard_data.loc[idex1,'ca_score'] = smart_guard_data.loc[idex1,'ca_score'] - 1
                        smart_guard_data.loc[idex2,'ca_score'] = smart_guard_data.loc[idex2,'ca_score'] - 1
        if s3 == 'nan':
            smart_guard_data.loc[idex1,'pa_score'] = -100
        else:
            for j in range( i+1 , len(filterd_data.index) ):
                # print('%d=>%d'%(j,filterd_data.index[j]))
                idex2 = filterd_data.index[j]
                s4 = smart_guard_data.loc[idex2,'preventive_action']
                t2 = smart_guard_data.loc[idex2,'event_date']
                sc2 = tf_similarity(s3, s4)
                #如果時間差在2年內
                if abs(t1 -t2) <pd.Timedelta(days = 730):       
                    #且如果相似度高於 ? %
                    if(sc2 >0.80):
                        # print('**PA: %d<==>%d  案件時間差 =%d 相似度%.3f'%(filterd_data.index[i],filterd_data.index[j],abs(t1-t2).days,sc2))
                        smart_guard_data.loc[idex1,'pa_score'] = smart_guard_data.loc[idex1,'pa_score'] - 1
                        smart_guard_data.loc[idex2,'pa_score'] = smart_guard_data.loc[idex2,'pa_score'] - 1
    return smart_guard_data

R_C_list = ['CAPA effectiveness', 'Competency/Workload', 'Document/Spec gap', 'Equipment/tool/Fixture Mgmt', 
'ESD', 'Human discipline', 'Infrastructure/Environment control/5S', 'Training and certificate', 
'Leadership engagement', 'Materials management', 'Mfg process design', 'Parameter setting/control', 
'Product HW design', 'Product SW design', 'Program/Script control', 'Record management', 
'SOP discipline', 'Supplier quality', 'System tool', 'Timely communication/Escalation' ]
R_C_list = [item.lower() for item in R_C_list]


def calaulate_similarity_prepare(smart_guard_data):
    ##str.lower()  => 轉換小寫
    smart_guard_data['corrective_action'] = smart_guard_data['corrective_action'].astype(str).str.lower()
    smart_guard_data['preventive_action'] = smart_guard_data['preventive_action'].astype(str).str.lower()
    smart_guard_data['rc_category_final2'] = smart_guard_data['rc_category_final2'].astype(str).str.lower()
    smart_guard_data['event_date'] = pd.to_datetime(smart_guard_data['event_date'])

    smart_guard_data.loc[:,'ca_score']=0
    smart_guard_data.loc[:,'pa_score']=0

    smart_guard_data.loc[:,'ca_score'] = -100
    smart_guard_data.loc[:,'pa_score'] = -100
    # smart_guard_data.loc[:,'ca_supervisor_evaluation'] = 0
    # smart_guard_data.loc[:,'pa_supervisor_evaluation'] = 0

    #同 RC category 第一層
    for rc  in R_C_list:
        findingdata1 = smart_guard_data[smart_guard_data.rc_category_final2 ==rc]
        question_list = list(set(findingdata1.question)) #去除重複 
        #同questions   第二層
        for questions in question_list:
            findingdata2 = findingdata1[findingdata1.question == questions]
            # smart_guard_data[ (smart_guard_data.rc_category_final2 ==rc) * (smart_guard_data.question ==questions)].ca_score.fillna(0)
            smart_guard_data.loc[(smart_guard_data.rc_category_final2 ==rc) * (smart_guard_data.question ==questions),'ca_score'] = 0
            smart_guard_data.loc[(smart_guard_data.rc_category_final2 ==rc) * (smart_guard_data.question ==questions),'pa_score'] = 0
            smart_guard_data = calaulate_similarity_and_score(smart_guard_data,findingdata2)

    return smart_guard_data


# smart_guard_data.to_csv('RC_Category_20V_test3_CAPA_score.csv',index =False, encoding = 'utf_8_sig')




