#! -*- coding: utf-8 -*-
import unittest
from tweet2vec import interface
import os
import shutil

class TestInterface(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass
        if 'tests' in os.path.dirname(__file__):
            cls.path_training_data = '../misc/trainer_example.txt'
            cls.path_validation_data = '../misc/trainer_example.txt'
            cls.path_test_data = '../misc/tester_example.txt'
            cls.model_dir = '../model'
            cls.model_jp_model = 'resources/jp_model'
        else:
            cls.path_training_data = 'misc/trainer_example.txt'
            cls.path_validation_data = 'misc/trainer_example.txt'
            cls.path_test_data = 'misc/tester_example.txt'
            cls.model_dir = 'model'
            cls.model_jp_model = 'tests/resources/jp_model'
        if not os.path.exists(cls.model_dir):
            os.mkdir(cls.model_dir)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.model_dir):
            shutil.rmtree(cls.model_dir)

    def test_init(self):
        tweet2vec_interface = interface.Twee2vecInterface()

    def generate_generic_input(self):
        """It generates generic input object with Japanese tweet"""
        seq_training_record = [
            interface.PostRecordObject(post_id=1, post_label='ノンストップ', post_text='ウナ刺し‼︎美味しそう‼︎食べいこ‼︎'),
            interface.PostRecordObject(post_id=2, post_label='ノンストップ', post_text='ウナギの刺身…どんな味なんだろう'),
            interface.PostRecordObject(post_id=3, post_label='ノンストップ', post_text='うなぎの刺し身。食べてみたいなぁ'),
            interface.PostRecordObject(post_id=4, post_label='ノンストップ', post_text='うなぎ…八百徳じゃないんだΣ（ﾟдﾟlll）魚魚一か〜'),
            interface.PostRecordObject(post_id=5, post_label='あなたの取説つくったー', post_text="""みさの正しい取扱い方法
(1)大きな声を出されると驚きます
(2)優しくされるとすぐに懐きます。
(3)お腹がすいたら機嫌が悪くなります。定期的に美味しいものを食べさせましょう
#あなたの取説つくったー
shindanmaker.com/714187

間違ってない"""),
            interface.PostRecordObject(post_id=6, post_label='あなたの取説つくったー', post_text="""なぎの正しい取扱い方法
(1)たまに旅に出たりします
(2)機嫌が悪い時は甘いものを与えましょう
(3)さみしがりやです。積極的に連絡してあげましょう
#あなたの取説つくったー
shindanmaker.com/714187
(´･_･`)"""),
            interface.PostRecordObject(post_id=7, post_label='あなたの取説つくったー', post_text="""るったんの正しい取扱い方法
(1)いじけていたら根気強く話を聞いてあげてください
(2)大きな声を出されると驚きます
(3)さりげないボディタッチが有効です。
#あなたの取説つくったー
shindanmaker.com/714187

3番確かに有効です。
"""),
            interface.PostRecordObject(post_id=8, post_label='あなたの取説つくったー', post_text="""眞杳ねみの正しい取扱い方法
(1)既読スルーにめっぽう弱いです
(2)1人で寝るのを嫌がります。ぬいぐるみでも与えて起きましょう
(3)褒めてあげるとよく伸びます
#あなたの取説つくったー
shindanmaker.com/714187

わかりやしたか。"""),
            interface.PostRecordObject(post_id=9, post_label='ノンストップ', post_text='ウナギの刺身なんて初めて見たな。'),
            interface.PostRecordObject(post_id=10, post_label='ノンストップ', post_text='うなぎの刺身初めて聞いた'),
            interface.PostRecordObject(post_id=11, post_label='ノンストップ', post_text="""シャフトの正しい取扱い方法
(1)いじけていたら根気強く話を聞いてあげてください
(2)隅っこが大好きです。
(3)押しに弱いので積極的に行きましょう
#あなたの取説つくったー
shindanmaker.com/714187
あぁ〜〜、シャフトっぽい。そして可愛い"""),
            interface.PostRecordObject(post_id=12, post_label='ノンストップ', post_text="""こよしの正しい取扱い方法
(1)いじけていたら根気強く話を聞いてあげてください
(2)たくさんリプを返されると喜びます
(3)大きな声を出されると驚きます
#あなたの取説つくったー
shindanmaker.com/714187
くそリプ歓迎""")]

        seq_test_record = [
            interface.PostRecordObject(post_id=13, post_text="""レイの正しい取扱い方法
(1)さりげないボディタッチが有効です。
(2)いじけていたら根気強く話を聞いてあげてください
(3)さみしがりやです。積極的に連絡してあげましょう
#あなたの取説つくったー
shindanmaker.com/714187
|･ω･｀)""", post_label=None),
            interface.PostRecordObject(post_id=14, post_text="""ふぉる花の正しい取扱い方法
(1)アニメを与えておけばひとりで遊びます
(2)なでなですると喜びます
(3)人前に出ると緊張します
#あなたの取説つくったー
shindanmaker.com/714187
まさしくその通り""", post_label=None),
            interface.PostRecordObject(post_id=15, post_text="うなぎの刺し身は、蒲焼きにするより美味しい", post_label=None),
            interface.PostRecordObject(post_id=16, post_text="ウナギのお刺身って見かけませんよね・・・調べてみたら、そこには意外な理由が。なんとウナギには毒があったのです！", post_label=None)
        ]

        training_dataset = interface.InputDataset.load_from_generic_input(seq_training_record)
        test_dataset = interface.InputDataset.load_from_generic_input(seq_test_record)

        return training_dataset, test_dataset


    def test_train(self):
        training_dataset = interface.InputDataset.load_table_data(path_to_data=self.path_training_data, separator='\t')
        validation_dataset = interface.InputDataset.load_table_data(path_to_data=self.path_validation_data, separator='\t')
        tweet2vec_interface = interface.Twee2vecInterface()
        tweet2vec_interface.train(
            training_dataset=training_dataset,
            validation_dataset=validation_dataset,
            is_use_trained_model=False,
            save_dir=self.model_dir)

    def test_predict(self):
        self.test_train()
        test_dataset = interface.InputDataset.load_table_data(path_to_data=self.path_test_data, separator='\t')
        model_object = interface.ModelObject.load_model(model_dir=self.model_dir)

        tweet2vec_interface = interface.Twee2vecInterface()
        tweet2vec_interface.predict(
            test_data=test_dataset,
            model_object=model_object)

    def test_generic_input(self):
        training_dataset, test_dataset = self.generate_generic_input()
        if not os.path.exists(self.model_jp_model):
            os.mkdir(self.model_jp_model)
        tweet2vec_interface = interface.Twee2vecInterface()
        ## training ##
        model_object = tweet2vec_interface.train(
            training_dataset=training_dataset,
            validation_dataset=training_dataset,
            is_use_trained_model=False,
            save_dir=self.model_jp_model)
        ## prediction ##
        seq_prediction_result = tweet2vec_interface.predict(
            test_data=test_dataset,
            model_object=model_object)
        for prediction_result in seq_prediction_result:
            self.assertTrue(isinstance(prediction_result, interface.PredictionRecordObject))
            print('Prediction result for id={} text={}, predicted-label={}'.format(prediction_result.post_id, prediction_result.post_text, prediction_result.prediction_label))

        import glob
        for model_file in glob.glob(os.path.join(self.model_jp_model, '*')):
            os.remove(model_file)


if __name__ == '__main__':
    unittest.main()