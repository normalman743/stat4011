import requests
import urllib3
from urllib.parse import urlparse, parse_qs
import pandas as pd
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def upload_file(csv_file_path, group_id=12507):
    """
    上传CSV文件到网站获取F1分数
    
    Args:
        csv_file_path (str): CSV文件路径
        group_id (int): 组ID，默认12507
    
    Returns:
        float: F1分数，如果失败返回None
    """
    
    url = "https://stat4011-part1.sta.cuhk.edu.hk/upload"
    
    try:
        print(f"正在上传文件: {csv_file_path}")
        
        with open(csv_file_path, 'rb') as f:
            files = {'submission': f}
            data = {'group_id': group_id}
            
            response = requests.post(
                url, 
                files=files, 
                data=data, 
                allow_redirects=False, 
                verify=False,
                timeout=30
            )
        
        if response.status_code == 302:
            redirect_url = response.headers.get('Location')
            parsed_url = urlparse(redirect_url)
            params = parse_qs(parsed_url.query)
            
            if 'score' in params:
                f1_score = float(params['score'][0])
                print(f"✅ 上传成功，F1分数: {f1_score:.6f}")
                return f1_score
            else:
                print("❌ 响应中未找到分数信息")
                return None
        else:
            print(f"❌ 上传失败，状态码: {response.status_code}")
            print(f"响应内容: {response.text}")
            return None
            
    except FileNotFoundError:
        print(f"❌ 文件未找到: {csv_file_path}")
        return None
    except requests.exceptions.Timeout:
        print("❌ 请求超时")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ 网络请求错误: {e}")
        return None
    except Exception as e:
        print(f"❌ 上传错误: {e}")
        return None

def validate_csv_format(csv_file_path):
    """
    验证CSV文件格式是否正确
    
    Args:
        csv_file_path (str): CSV文件路径
    
    Returns:
        bool: 格式是否正确
    """
    
    try:
        
        df = pd.read_csv(csv_file_path)
        
        # 检查列名
        required_columns = ['ID', 'Predict']
        if not all(col in df.columns for col in required_columns):
            print(f"❌ 缺少必要列: {required_columns}")
            return False
        
        # 检查Predict列值
        unique_values = df['Predict'].unique()
        if not all(val in [0, 1] for val in unique_values):
            print(f"❌ Predict列包含非法值: {unique_values}")
            return False
        
        print(f"✅ CSV格式验证通过")
        print(f"  账户数: {len(df)}")
        print(f"  预测Bad: {len(df[df['Predict'] == 1])}")
        print(f"  预测Good: {len(df[df['Predict'] == 0])}")
        
        return True
        
    except Exception as e:
        print(f"❌ CSV格式验证失败: {e}")
        return False

if __name__ == "__main__":
    print("=== 文件上传模块测试 ===")
    
    # 测试文件路径
    test_file = "/Users/mannormal/Desktop/课程/y4t1/stat 4011/part1/Qi Zihan/v7/FINAL_BEST.csv"
    
    # 验证格式
    if validate_csv_format(test_file):
        # 上传文件
        result = upload_file(test_file)
        print(f"最终结果: {result}")
    else:
        print("文件格式验证失败，跳过上传")