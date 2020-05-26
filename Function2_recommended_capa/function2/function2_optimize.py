#!/usr/bin/env python
# coding: utf-8
import os
import sys
import numpy as np
import pandas as pd 
import warnings
warnings.filterwarnings('ignore')
from .utils.opencc_self import OpenCC

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)

jieba_dist = os.path.join(BASE_DIR, 'data/dict.txt')
jieba_dist_2 = os.path.join(BASE_DIR, 'data/idf.txt')
import jieba
jieba.set_dictionary(jieba_dist)
jieba.initialize()
import jieba.analyse
jieba.analyse.set_idf_path(jieba_dist_2)

# opencc 位置
c_DIR = os.path.join(BASE_DIR, 'data/opencc/config')
d_DIR = os.path.join(BASE_DIR, 'data/opencc/dictionary')
cc = OpenCC('s2tw', CONFIG_DIR = c_DIR, DICT_DIR = d_DIR)

###載入關鍵詞庫
feature_dir = os.path.join(BASE_DIR, 'data/RC_Category_2020_feature_word_to_low_vr2.xlsx')
sg_feature_words = pd.read_excel (feature_dir)  
# 取得關鍵詞表 建成list
sg_feature_words['feature_words']
feature_list = ','.join(sg_feature_words['feature_words'])
feature_list = feature_list.split(',')

try:
    bool(type(unicode))
except NameError:
    unicode = str

def capa_rank_calculate(smart_guard_data, text_details, input_category, question, mode):
	#轉小寫
	smart_guard_data['rc_category_final2'] = smart_guard_data['rc_category_final2'].astype(str).str.lower()
	input_category = input_category.lower()
	#簡轉繁
	text_details = cc.convert(text_details)
	#根據敘述 抓出關鍵字!!!        
	text_list = jieba.lcut(str(text_details), cut_all=False, HMM = True)
	text_list = [x for x in text_list if x != ' ']
	text_list = list(set(text_list)) #去除重複 
	key_word_list = []
	# print('推薦問題敘述中的關鍵字:')
	for word in text_list:  
	    if word in  feature_list:     #依據關鍵詞表篩選
	        # print(word,end=' ')
	        key_word_list.append(word)#!!!!!! 輸出1. 找到的關鍵字 list

	data_range = smart_guard_data[smart_guard_data['rc_category_final2']==input_category]
	data_range = data_range.fillna('0')#辭典空資料補0
	data_range = data_range[data_range.ca_score != -100]
	#算評鑑總分 
	data_range['total_ca_score'] = data_range['ca_score'] + data_range['ca_supervisor_evaluation']
	data_range['total_pa_score'] = data_range['pa_score'] + data_range['pa_supervisor_evaluation']
	#######################################        排序       ################################################

	### 關鍵詞 Rank 
	keyword_score_c = np.zeros(len(data_range)).tolist()
	keyword_score_p = np.zeros(len(data_range)).tolist()
	state_f = []
	state_r = []
	state_c = []
	state_p = []
	#計算關鍵詞積分
	for words  in  key_word_list :
	    state_f = data_range['finds_detail'].str.contains(words)*1
	    state_r = data_range['root_cause'].str.contains(words)*1
	    state_c = data_range['corrective_action'].str.contains(words)*1
	    state_p = data_range['preventive_action'].str.contains(words)*1
	    keyword_score_c = keyword_score_c + state_f + state_c + state_r    #ca 積分加總
	    keyword_score_p = keyword_score_p + state_f + state_r + state_p


	####預算分隔分數同RC下:
	grouped = data_range.groupby(['question'])
	data_range['same_Q'] = data_range.question == question

	#final排序 結合評鑑分數和關鍵詞相關程度排序:
	if mode =='ca':
		###CA   rank
		data_range['Mid_ca_score'] = grouped['total_ca_score'].transform(lambda x: (max(x) + min(x))/2)
		data_range['Q_ca_score_High'] = data_range['total_ca_score']>data_range['Mid_ca_score']
		data_range['keyword_score_ca'] = keyword_score_c
		#phase1   same_Q = True   Q_ca_score_High = True
		phase12_data = data_range[data_range['same_Q']==True]
		phase1_ca_data = phase12_data[phase12_data['Q_ca_score_High']==True]
		p1_ca_rank = phase1_ca_data.keyword_score_ca.sort_values(ascending=False)
		#phase2   same_Q = True   Q_ca_score_High = False
		phase2_ca_data = phase12_data[phase12_data['Q_ca_score_High']==False]
		p2_ca_rank = phase2_ca_data.keyword_score_ca.sort_values(ascending=False)
		#phase3   same_Q = Fasle   Q_ca_score_High = True
		phase34_data = data_range[data_range['same_Q']==False]
		phase3_ca_data = phase34_data[phase34_data['Q_ca_score_High']==True]
		p3_ca_rank = phase3_ca_data.keyword_score_ca.sort_values(ascending=False)
		#phase4   same_Q = Fasle   Q_ca_score_High = Fasle
		phase4_ca_data = phase34_data[phase34_data['Q_ca_score_High']==False]
		p4_ca_rank = phase4_ca_data.keyword_score_ca.sort_values(ascending=False)
		ca_rank_list =  list(p1_ca_rank.index) + list(p2_ca_rank.index) + list(p3_ca_rank.index) + list(p4_ca_rank.index)
		recommend_index_list = ca_rank_list

	else:
		###PA   rank
		data_range['Mid_pa_score'] = grouped['total_pa_score'].transform(lambda x: (max(x) + min(x))/2)
		data_range['Q_pa_score_High'] = data_range['total_pa_score']>data_range['Mid_pa_score']
		data_range['keyword_score_pa'] = keyword_score_p
		phase12_data = data_range[data_range['same_Q']==True]
		phase1_pa_data = phase12_data[phase12_data['Q_pa_score_High']==True]
		p1_pa_rank = phase1_pa_data.keyword_score_pa.sort_values(ascending=False)
		phase2_pa_data = phase12_data[phase12_data['Q_pa_score_High']==False]
		p2_pa_rank = phase2_pa_data.keyword_score_pa.sort_values(ascending=False)
		phase34_data = data_range[data_range['same_Q']==False]
		phase3_pa_data = phase34_data[phase34_data['Q_pa_score_High']==True]
		p3_pa_rank = phase3_pa_data.keyword_score_pa.sort_values(ascending=False)
		phase4_pa_data = phase34_data[phase34_data['Q_pa_score_High']==False]
		p4_pa_rank = phase4_pa_data.keyword_score_pa.sort_values(ascending=False)
		pa_rank_list =  list(p1_pa_rank.index) + list(p2_pa_rank.index) + list(p3_pa_rank.index) + list(p4_pa_rank.index)
		recommend_index_list = pa_rank_list

	if mode =='ca':
		recommend_list = list(smart_guard_data.iloc[recommend_index_list,:].corrective_action)
	else:
		recommend_list = list(smart_guard_data.iloc[recommend_index_list,:].preventive_action)
	return key_word_list, recommend_list
	