# for pytest

from flask_testing import TestCase

from app import app
import config


class BaseTestCase(TestCase):

    def create_app(self):
		# 必要。須回傳 Flask 實體。
        app.config.from_object(config.TestingConfig)
        return app

    def setUp(self):
		# 可不寫。測試前會執行的東西，相當於 pytest 中 @pytest.fixture 這個裝飾器
		# 可以用於生出一個乾淨(沒有資料)的資料庫之類的，不過因為我是用奇怪的方式弄出類似資料庫的東東，所以就沒有寫
        pass

    def tearDown(self):
		# 可不寫。測試後會執行的東西，相當於 pytest 中 @pytest.fixture 這個裝飾器 function 內 yield 之後的程式
		# 可以用於刪除不乾淨(測試後被塞入資料)的資料庫之類的
        pass
        
    @classmethod
    def setUpClass(self):
		# 可不寫。相當於 setUp ，不過不同於 setUp 是執行一個 Function ，而是先執行一個 Class，詳細用法參考 @classmethod 或是下面的網址
		# https://docs.python.org/zh-tw/3/library/unittest.html#unittest.TestCase.setUpClass
        pass
        
    @classmethod
    def tearDownClass(self):
		# 可不寫。相當於 tearDown ，不過 setUpClass 同樣為執行一個 Class，詳細用法參考 @classmethod 或是下面的網址
		# https://docs.python.org/zh-tw/3/library/unittest.html#unittest.TestCase.tearDownClass
        pass
        
    def setUpModule():
		# 可不寫。同樣相當於 setUp ，不過不同於 setUp 以及 setUpClass 是執行一個 Function 或是 Class，而是先執行一個 Module，詳細用法參考下面的網址
		# https://docs.python.org/zh-tw/3/library/unittest.html#setupmodule-and-teardownmodule
        pass
        
    def tearDownModule():
		# 可不寫。相當於 tearDown ，不過 setUpModule 同樣為執行一個 Module，詳細用法參考 setUpModule 的網址
        pass