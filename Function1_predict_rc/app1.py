# coding=utf-8
import os
from flask import request, url_for
from flask_api import FlaskAPI
from flask_api.renderers import JSONRenderer, HTMLRenderer
from flask_api.decorators import set_renderers

# 統一在這邊引用各個 ai model
from function1 import function1


# 可參考網站 https://www.flaskapi.org/
app = FlaskAPI(__name__)

# 模型初始化: 請將所有model在這邊統一呼叫
rc_category_model = function1.CnnModel()


@app.route('/')
@set_renderers(HTMLRenderer)
def index():
    return '<html><body><h1>This is Flask app for AI model!</h1></body></html>'
#aaaa

@app.route('/rc_category_pred', methods=['POST'])
def rc_category_predict():
    text_details = request.data.get('find_detail', '')
    text_details = function1.cc.convert(text_details)
    first, top5 = rc_category_model.predict(text_details)
    toplist = top5.rc_category.tolist()
    return {'top5': toplist}


if __name__ == "__main__":

    if os.name == "nt":
        app.run(port=8002, debug=True, threaded=True)
    else:
        app.run(host="0.0.0.0", port=8002, debug=True, threaded=True)
