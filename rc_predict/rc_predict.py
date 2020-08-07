import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.append(BASE_DIR)
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.keras as kr
from utils.text_model import TEXTConfig,TextCNN
from data.cnews_loader import read_category, read_vocab
from utils.opencc_self import OpenCC

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 設置cpu執行
try:
    bool(type(unicode))
except NameError:
    unicode = str


print(BASE_DIR)
# 載入模型的辭典
vocab_dir = 'data/rc_category_data'
vocab_path = os.path.join(BASE_DIR, vocab_dir, 'cnews.vocab.txt')  # vocabulary.txt
# 載入分類模型
save_dir = 'rc_predict/checkpoints/textcnn'
save_path = os.path.join(BASE_DIR, save_dir, 'best_validation')  # 最佳验证结果保存路径

# opencc 位置
c_DIR = os.path.join(BASE_DIR, 'data/opencc/config')
d_DIR = os.path.join(BASE_DIR, 'data/opencc/dictionary')


# RC category列表 共20類
# RC_category = [
    #     'CAPA effectiveness',
    #     'Competency/Workload',
    #     'Document/Spec gap',
    #     'Equipment/tool/Fixture Mgmt',
    #     'ESD',
    #     'Human discipline',
    #     'Infrastructure/Environment control/5S',
    #     'Training and certificate',
    #     'Leadership engagement',
    #     'Materials management',
    #     'Mfg process design',
    #     'Parameter setting/control',
    #     'Product HW design',
    #     'Product SW design',
    #     'Program/Script control',
    #     'Record management',
    #     'SOP discipline',
    #     'Supplier quality',
    #     'System tool',
    #     'Timely communication/Escalation',
    # ]
# RC_category = [item.lower() for item in RC_category]  # 轉小寫

RC_category = [
        'human discipline|人員紀律問題',
        'equipment/tool/fixture mgmt|設備/治工具管理',
        'supplier quality|供應商品質問題',
        'sop discipline|SOP製作問題',
        'materials management|物料管理',
        'infrastructure/environment control/5s|基礎設施/環境控制/5S',
        'system tool|系統工具',
        'record management|記錄管理',
        'esd|ESD相關問題',
        'document/spec gap|文件/規格差異',
        'mfg process design|製造工藝設計',
        'training and certificate|培訓與認證',
        'program/script control|程序/腳本控制',
        'parameter setting/control|參數設置/控制',
        'capa effectiveness|對策有效性',
        'leadership engagement|領導參與',
        'product hw design|產品硬件設計問題',
        'timely communication/escalation|及時溝通/上報',
        'competency/workload|能力/工作量',
        'product sw design|產品軟件設計問題',
    ]
RC_category = [ x for x in RC_category]
RC_category_id = dict(zip(RC_category, range(len(RC_category))))


class CnnModel:
    def __init__(self):
        self.config = TEXTConfig()
        self.categories, self.cat_to_id = read_category()
        self.words, self.word_to_id = read_vocab(vocab_path)
        self.config.vocab_size = len(self.words)
        self.model = TextCNN(self.config)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=save_path)  # 读取保存的模型

    def predict(self, message):
        # 支持不论在python2还是python3下训练的模型都可以在2或者3的环境下运行
        content = unicode(message)
        data = [self.word_to_id[x] for x in content if x in self.word_to_id]

        feed_dict = {
            self.model.input_x: kr.preprocessing.sequence.pad_sequences([data], self.config.seq_length),
            self.model.keep_prob: 1.0,
        }

        y_pred_cls = self.session.run(self.model.y_pred_cls, feed_dict=feed_dict)
        score_matrix = self.session.run(self.model.y_pred_matrix, feed_dict=feed_dict)

        acc_data = pd.DataFrame(index=['rank1', 'rank2', 'rank3', 'rank4', 'rank5'], columns=['rc_category'])
        acc_data.loc['rank1', 'rc_category'] = RC_category[(score_matrix[0]).argmax()]
        score_matrix[0, (score_matrix[0]).argmax()] = np.min(score_matrix[0]) - 100
        acc_data.loc['rank2', 'rc_category'] = RC_category[(score_matrix[0]).argmax()]
        score_matrix[0, (score_matrix[0]).argmax()] = np.min(score_matrix[0]) - 100
        acc_data.loc['rank3', 'rc_category'] = RC_category[(score_matrix[0]).argmax()]
        score_matrix[0, (score_matrix[0]).argmax()] = np.min(score_matrix[0]) - 100
        acc_data.loc['rank4', 'rc_category'] = RC_category[(score_matrix[0]).argmax()]
        score_matrix[0, (score_matrix[0]).argmax()] = np.min(score_matrix[0]) - 100
        acc_data.loc['rank5', 'rc_category'] = RC_category[(score_matrix[0]).argmax()]

        return self.categories[y_pred_cls[0]], acc_data


# 模型初始化     需要一段時間    啟動後可以持續進行predict
# model = CnnModel()
# # 簡轉繁
cc = OpenCC('s2twp', CONFIG_DIR=c_DIR, DICT_DIR=d_DIR)


######################################     input     ##########################################
# text_details = input("輸入問題敘述= (請將Finds detail 、 Root cause 合併輸入 ,文句中勿輸入enter!!)\n")
# text_details =cc.convert(text_details)
# ##################################        模型預測    ##########################################
# first, top5 = model.predict(text_details)  # rc_category_predict
# toplist = top5.rc_category.tolist()


# print('預測結果: ')
# k = 1
# for i in toplist:
#     print('Rank %d: %s' % (k, i))
#     k = k + 1
