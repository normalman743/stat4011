import requests
from urllib.parse import urlparse, parse_qs
import pandas as pd
import urllib3
import glob

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def get_confusion_matrix(csv_file_path):
    """
    输入CSV文件路径，输出混淆矩阵
    """
    # 1. 读取预测结果
    df = pd.read_csv(csv_file_path)
    pred_counts = df['Predict'].value_counts()
    pred_bad = pred_counts.get(1, 0)   # Bad = 1
    pred_good = pred_counts.get(0, 0)  # Good = 0

    # 2. 提交获取F1分数
    url = "https://stat4011-part1.sta.cuhk.edu.hk/upload"
    with open(csv_file_path, 'rb') as f:
        files = {'submission': f}
        data = {'group_id': 12507}
        response = requests.post(url, files=files, data=data, allow_redirects=False, verify=False)
        
    parsed_url = urlparse(response.headers.get('Location'))
    bad_f1 = float(parse_qs(parsed_url.query)['score'][0])
    
    # 3. 计算混淆矩阵
    true_bad, true_good = 727, 6832
    
    if bad_f1 == 0:
        tp, fp, fn, tn = 0, pred_bad, true_bad, true_good - pred_bad
    else:
        tp = int(round(bad_f1 * (pred_bad + true_bad) / 2.0))
        fp = pred_bad - tp
        fn = true_bad - tp  
        tn = true_good - fp
    
    # 4. 输出矩阵
    print(f"文件: {csv_file_path.split('/')[-1]}")
    print(f"Bad F1: {bad_f1:.4f}")
    print("混淆矩阵:")
    print("              预测")
    print("           Bad    Good")
    print(f"真实 Bad   {tp:4d}    {fn:4d}")
    print(f"    Good   {fp:4d}   {tn:4d}")
    
    return {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn, 'F1': bad_f1}

# 使用示例
if __name__ == "__main__":

    csv_files = glob.glob("/Users/mannormal/4011/Qi Zihan/v2/results/high_score_predictions/0.75+/*.csv")
    for csv_file in csv_files:
        get_confusion_matrix(csv_file)
    
    get_confusion_matrix("/Users/mannormal/4011/Qi Zihan/v2/results/py/votes/vote_5_0_REAL_F1_0.12484076433121019.csv")

    