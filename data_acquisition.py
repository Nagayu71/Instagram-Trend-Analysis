import requests
import datetime
import json
import time

class InstagramHashtagSearch:    
    # fieldsパラメーターを指定
    fields = "timestamp,caption,id"
    
    #  コンストラクタ引数：インスタグラムのビジネスアカウントID、APIのトークン. この2つの情報を持ったインスタンスを生成する.それに対してメソッドを作用させる
    def __init__(self, business_account_id, access_token):
        # インスタンス変数（メンバ変数）の設定
        self.business_account_id= business_account_id
        self.access_token = access_token
        
    
    # ハッシュタグIDを取得
    def get_hashtag_id(self, hashtag_query):
        # ハッシュタグID取得用URL
        ig_hashtagID_search_api = f"https://graph.facebook.com/ig_hashtag_search?user_id={self.business_account_id}&q={hashtag_query}&access_token={self.access_token}"
        # レスポンスをPythonが扱うことのできる対象（JSON）に変換し、IDを取得
        hash_id = requests.get(ig_hashtagID_search_api).json()['data'][0]['id']
        return hash_id
    
    # APIを叩いて見た目JSON形式のデータを取得
    def request_media(self, hashtag_query, num):
        # ハッシュタグID取得
        hash_id = self.get_hashtag_id(hashtag_query)
        
        # 取得数に応じて条件分岐
        if num == 50:
            hashtag_search_api = f"https://graph.facebook.com/{hash_id}/recent_media?user_id={self.business_account_id}&fields={self.fields}&access_token={self.access_token}&limit={str(num)}"
            # レスポンスをPythonが扱うことのできる対象（JSON）に変換
            res_original = requests.get(hashtag_search_api).json()
            assert "data" in res_original, 'There must be an error. Your access token might be invalid'
            res_data = res_original['data']
        else:
            hashtag_search_api = f"https://graph.facebook.com/{hash_id}/recent_media?user_id={self.business_account_id}&fields={self.fields}&access_token={self.access_token}&limit={str(num)}"
            # レスポンスをPythonが扱うことのできる対象（JSON）に変換
            res_original = requests.get(hashtag_search_api).json()
            assert "data" in res_original, 'There must be an error. Your access token might be invalid'
            res_data = res_original['data']
            # APIの仕様上、一度のリクエストで取得できるのは最大50件までなので、それ以上取得したい場合は2回リクエストを送る
            next_50_url = res_original['paging']['next']
            next_50 = requests.get(next_50_url).json()
            assert "data" in next_50, 'There must be an error. Your access token might be invalid'
            
            for data in next_50['data']:
                res_data.append(data)
        return res_data

# 取得したビジネスアカウントIDとアクセストークン
business_account_id = '*****************'
access_token = '*****************'

# 15個のハッシュタグで最新投稿を検索
query_list = ['神戸', 'ダイエット', 'スターバックス', 'スイーツ', 'ワールドカップ', 'コーデ', '紅葉', 'キャンペーン', 'アニメ', '関西旅行', 'universitylife', 'チーズ', 'ええじゃろ広島の秋2022', 'じゃけぇ広島に恋しとる', '新海誠'] 
querys = ['kobe', 'diet', 'starbucks', 'sweets', 'worldcup', 'coordinate', 'momiji', 'campaign', 'anime', 'kansairyoko', 'universitylife', 'cheese', 'hiroshima_jpn', 'explore_hiroshima', 'shinkaimakoto']

created_object = InstagramHashtagSearch(business_account_id, access_token)

for query, fname in zip(query_list, querys):
    # 取得した時間（JST）を記録
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
    tstr = now.strftime('%Y-%m-%d_%H_%M')
    
    # リクエスト送信
    api_res = created_object.request_media(query, 100)
    
    # ファイルネームを可変にする(yodaのなかでの絶対パスにしておく)
    with open(f"/{fname}/{fname}_{tstr}.txt", mode='w') as f:
        json.dump(api_res, f, indent=4)
    
    time.sleep(1)