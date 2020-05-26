# coding=utf-8
import os
from flask import request, url_for
from flask_api import FlaskAPI
from flask_api.renderers import JSONRenderer, HTMLRenderer
from flask_api.decorators import set_renderers
import pandas as pd 
# 統一在這邊引用各個 ai model
from function2 import function2_optimize
from function2.function2_optimize import capa_rank_calculate


# 可參考網站 https://www.flaskapi.org/
app = FlaskAPI(__name__)

# 模型初始化: 請將所有model在這邊統一呼叫



@app.route('/')
@set_renderers(HTMLRenderer)
def index():
    return '<html><body><h1>This is Flask app for AI model!</h1></body></html>'


@app.route('/capa_recommend', methods=['POST'])
def capa_recommend():
    # smart_guard_data = ####須從database 取得資料
    # smart_guard_data = pd.read_csv ("RC_Category_20V_04_27_CAPA_score.csv")  測試用

    text_details = request.data.get('find_detail','')
    text_details = function2_optimize.cc.convert(text_details)
    input_category = request.data.get('rc_category')
    mode = request.data.get('choose_mode')   ##只接'ca'、'pa'兩種

    question = '0.0.0'     #request.data.get('question', '')現階段不開放
    input_category=eval(input_category)#去除引號
    mode=eval(mode)#去除引號
    key_word_list, recommend_list = capa_rank_calculate(smart_guard_data, text_details, input_category, question, mode)


    print('text=',text_details)
    print('rc=',input_category)
    print('mode=',mode)

    # print(key_word_list)
    # print(recommend_index_list)
    # return {'1': text_details,'2':input_category,'3':mode}
    return {'keyword_list': key_word_list,'text_list':recommend_list}


if __name__ == "__main__":
    app.config['JSON_AS_ASCII'] = False

    if os.name == "nt":
        app.run(host="0.0.0.0", port=8002, debug=True, threaded=True)
    else:
        app.run(host="0.0.0.0", port=8002, debug=True, threaded=True)
