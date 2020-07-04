# coding=utf-8
import os
import sys
from flask import request, url_for
from flask_api import FlaskAPI
from flask_api.renderers import JSONRenderer, HTMLRenderer
from flask_api.decorators import set_renderers
import pandas as pd

# 統一在這邊引用各個 ai model
from rc_predict import rc_predict
from capa_recommended import capa_recommended
from capa_recommended.capa_recommended import capa_rank_calculate
from capa_score.update_capa_score import calaulate_similarity_prepare


# 可參考網站 https://www.flaskapi.org/
app = FlaskAPI(__name__)

# 模型初始化: 請將所有model在這邊統一呼叫
rc_category_model = rc_predict.CnnModel()


@app.route('/')
@set_renderers(HTMLRenderer)
def index():
    return '<html><body><h1>This is Flask app for AI model!</h1></body></html>'


@app.route('/rc_category_pred', methods=['POST'])
def rc_category_predict():
    text_details = request.data.get('find_detail', '')
    text_details = rc_predict.cc.convert(text_details)
    first, top5 = rc_category_model.predict(text_details)
    toplist = top5.rc_category.tolist()
    return {'top5': toplist}


@app.route('/capa_recommend', methods=['POST'])
def capa_recommend():
    # smart_guard_data = ####須從database 取得資料
    # 需要欄位 'rc_category_final2'、'ca_score'、'pa_score'、'ca_supervisor_evaluation'、'pa_supervisor_evaluation'、
    #         'finds_detail'、'root_cause'、'corrective_action'、'preventive_action'、'question'
    smart_guard_data = pd.read_csv("RC_Category_20V_04_27_CAPA_score.csv")  # =>測試用

    text_details = request.data.get('find_detail', '')
    text_details = capa_recommended.cc.convert(text_details)
    input_category = request.data.get('rc_category')
    mode = request.data.get('choose_mode')  # 從前端得到訊號 只接'ca'、'pa'兩種

    question = '0.0.0'  # request.data.get('question', '')現階段不開放
    input_category = eval(input_category)  # 去除引號
    mode = eval(mode)  # 去除引號
    key_word_list, recommend_list = capa_rank_calculate(smart_guard_data, text_details, input_category, question, mode)

    return {'keyword_list': key_word_list, 'text_list': recommend_list}


@app.route('/capa_score_calclation', methods=['POST'])
def capa_score_calclation():
    # smart_guard_data = ####須從database 取得資料
    # 需要欄位 'rc_category_final2'、'ca_score'、'pa_score'、'event_date'、
    #          、'corrective_action'、'preventive_action'、'question'

    smart_guard_data = pd.read_csv("RC_Category_20V_04_27_CAPA_score.csv")  # 測試用
    print('Waiting for calculation...')
    calaulate_similarity_prepare(smart_guard_data)
    # smart_guard_data.to_csv('RC_Category_20V_test3_CAPA_score.csv',index =False, encoding = 'utf_8_sig') #測試用

    # 更新 database中的 'ca_score'、'pa_score'
    return {'ca_sore': list(smart_guard_data.ca_score), 'pa_sore': list(smart_guard_data.pa_score)}


if __name__ == "__main__":

    if os.name == "nt":
        app.run(port=8002, debug=True, threaded=True)
    else:
        app.run(host="0.0.0.0", port=8002, debug=True, threaded=True)
