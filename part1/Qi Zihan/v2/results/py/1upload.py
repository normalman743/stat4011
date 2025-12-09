import requests
from urllib.parse import urlparse, parse_qs
import urllib3
import os
from time import sleep
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def submit_file(group_id, csv_file_path):
    url = "https://stat4011-part1.sta.cuhk.edu.hk/upload"
    
    with open(csv_file_path, 'rb') as f:
        files = {'submission': f}
        data = {'group_id': group_id}
        
        response = requests.post(url, files=files, data=data, allow_redirects=False, verify=False)
        
        if response.status_code == 302:
            redirect_url = response.headers.get('Location')
            
            # 直接从URL参数提取分数
            parsed_url = urlparse(redirect_url)
            params = parse_qs(parsed_url.query)
            score = float(params['score'][0])
            
            return score
        else:
            return None

# 使用
if __name__ == "__main__":
    path = "/Users/mannormal/4011/Qi Zihan/v2/results/py/votes/"
    filenames = [f for f in os.listdir(path) if f.endswith('.csv')]
    for filename in filenames:
        csv_file = path + filename
        score = submit_file(12507, csv_file)
        print(f"File: {filename}, F1 Score: {score}")
        if score is not None:
            new_filename = filename.rsplit('.csv', 1)[0] + f"_REAL_F1_{score}.csv"
            new_filepath = os.path.join(path, new_filename)
            old_filepath = os.path.join(path, filename)
            os.rename(old_filepath, new_filepath)
            print(f"Renamed file: {old_filepath} -> {new_filepath}")
        else:
            print("Score not available, file not renamed.")
        sleep(1)  # 避免过快提交，防止服务器拒绝请求
    
    
    
    filename155 = "Transformer_basic_submission_BEST_FOLD_8_f1_0.8983_epochs_130.155.csv"
    csv_file155 = path + filename155

    submit_file(12507, csv_file155)
    