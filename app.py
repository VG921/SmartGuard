# coding=utf-8
import os
import sys
from flask import request, url_for
from flask_api import FlaskAPI
from flask_api.renderers import JSONRenderer, HTMLRenderer
from flask_api.decorators import set_renderers
from flask_apscheduler import APScheduler
import pandas as pd

# 統一在這邊引用各個 ai model
from rc_predict import rc_predict
from capa_recommended import capa_recommended
from capa_recommended.capa_recommended import capa_rank_calculate
from capa_score.update_capa_score import calaulate_similarity_prepare


# 可參考網站 https://www.flaskapi.org/
app = FlaskAPI(__name__)    # 例項化flask

# 模型初始化: 請將所有model在這邊統一呼叫
rc_category_model = rc_predict.CnnModel()

class Config(object):
    JOBS=[
        {
            'id':'job1',
            'func':'__main__:job_update_icar_data',
            'trigger':'cron',
            'hour':23,
            'minute':59
        }
    ]

def job_update_icar_data():   # 做定時任務的任務。
    # print(str(a)+' '+str(b))
    smart_guard_data = pd.read_csv("RC_Category_20V_04_27_CAPA_score.csv") 
    print('ori=',len(smart_guard_data))

    url1="http://10.129.8.17:802/Api/iCarHeatmapFol?status=%22Closed%22&quarter=%22%20%22"
    new_data1 = pd.read_json(url1)
    url2 = "http://10.129.8.17:802/Api/iCARStarscore"
    score_data = pd.read_json(url2)

    usedata = new_data1[~new_data1['rc_category'].isna()]
    usedata = usedata.merge(score_data,on='icar',how='left')
    usedata['event_id'] = usedata['icar']
    usedata = usedata.rename(columns = { 'bu':'BU','sub_section':'sub-section' ,
                                        'corrective':'ca_supervisor_evaluation',
                                        'prevention':'pa_supervisor_evaluation',
                                        'functions':'function',
                                        'ca_Implemented_date':'ca_implemented_date',
                                        'pa_implemented_date':'pa_implemented_date',
                                        'status':'audit_status','comp_pn':'comn_pn'})

    usedata = usedata.drop(columns=['section','area','rootcause','audit_type','owner','dueday','auditor','ca_category'])
    usedata['rc_category_final1'] = usedata['rc_category']
    usedata['rc_category_final2'] = usedata['rc_category']
    #資料合併
    smart_guard_data = pd.concat([smart_guard_data, usedata],sort=False).reset_index(drop=True)

    #篩選 需要修正的資料
    mask = smart_guard_data['question'].str.endswith('.')
    smart_guard_data.loc[mask,'question'] = smart_guard_data.loc[mask,'question'].str[:-1]
    smart_guard_data['ca_score'] = smart_guard_data['ca_score'].fillna(0)
    smart_guard_data['pa_score'] = smart_guard_data['pa_score'].fillna(0)
    smart_guard_data['ca_supervisor_evaluation'] = smart_guard_data['ca_supervisor_evaluation'].fillna(0)
    smart_guard_data['pa_supervisor_evaluation'] = smart_guard_data['pa_supervisor_evaluation'].fillna(0)
    #重新計算 相似度 對重複性高的CA PA做扣分
    smart_guard_data = calaulate_similarity_prepare(smart_guard_data)

    smart_guard_data.to_csv('RC_Category_21V_update_CAPA_score.csv',index =False, encoding = 'utf_8_sig') #
    print('data update finished!')


app.config.from_object(Config())# 為例項化的flask引入配置




@app.route('/')
@set_renderers(HTMLRenderer)
def index():
    return '<html><body><h1>This is Flask app for AI model!</h1></body></html>'


@app.route('/rc_category_pred', methods=['POST'])
def rc_category_predict():
    text_details = request.data.get('find_detail', '')
    text_details = rc_predict.cc.convert(text_details)
    text_details = text_details.lower()
    first, top5 = rc_category_model.predict(text_details)
    toplist = top5.rc_category.tolist()
    return {'top5': toplist}


@app.route('/capa_recommend', methods=['POST'])
def capa_recommend():
    # smart_guard_data = ####須從database 取得資料
    # 需要欄位 'rc_category_final2'、'ca_score'、'pa_score'、'ca_supervisor_evaluation'、'pa_supervisor_evaluation'、
    #         'finds_detail'、'root_cause'、'corrective_action'、'preventive_action'、'question'
    smart_guard_data = pd.read_csv("RC_Category_21V_update_CAPA_score.csv")  # =>測試用
    problem_capa = pd.read_excel("problem_type_capa.xlsx")

    text_details = request.data.get('find_detail', '')
    text_details = capa_recommended.cc.convert(text_details)
    input_category = request.data.get('rc_category')
    mode = request.data.get('choose_mode')  # 從前端得到訊號 只接'ca'、'pa'兩種
    problem_type =  request.data.get('problem_type') #10/12新增 problem type 設為最優先推薦依據

    question = '0.0.0'  # request.data.get('question', '')現階段不開放
    input_category = eval(input_category)  # 去除引號
    mode = eval(mode)  # 去除引號
    problem_type = eval(problem_type)
    ###多增加 回傳分數值 評鑑分數
    key_word_list, recommend_list, recommend_index = capa_rank_calculate(smart_guard_data, text_details, input_category, question, mode, problem_type, problem_capa)

    return {'keyword_list': key_word_list, 'text_list': recommend_list, 'text_index': recommend_index}


@app.route('/capa_score_calclation', methods=['POST'])
def capa_score_calclation():
    # smart_guard_data = ####須從database 取得資料
    # 需要欄位 'rc_category_final2'、'ca_score'、'pa_score'、'event_date'、
    #          、'corrective_action'、'preventive_action'、'question'
    #行時間約一分鐘
    smart_guard_data = pd.read_csv("RC_Category_21V_update_CAPA_score.csv")  # 測試用
    print('Waiting for calculation...')
    calaulate_similarity_prepare(smart_guard_data)
    
    #可直接儲存蓋掉原來的smart_guard_data
    # smart_guard_data.to_csv('RC_Category_20V_test3_CAPA_score.csv',index =False, encoding = 'utf_8_sig') #測試用

    # 更新 database中的 'ca_score'、'pa_score'
    return {'ca_sore': list(smart_guard_data.ca_score), 'pa_sore': list(smart_guard_data.pa_score)}


if __name__ == "__main__":

    scheduler=APScheduler()  # 例項化APScheduler
    scheduler.init_app(app)  # 把任務列表放進flask
    scheduler.start() # 啟動任務列表

    if os.name == "nt":
        app.run(port=8002, debug=True, threaded=True)
    else:
        app.run(host="0.0.0.0", port=8002, debug=True, threaded=True)
