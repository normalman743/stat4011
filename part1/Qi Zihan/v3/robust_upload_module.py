import requests
import urllib3
from urllib.parse import urlparse, parse_qs
from time import sleep
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def upload_file(csv_file_path, group_id=12507, max_retries=None):
    """
    上传CSV文件到网站获取F1分数（增强版：包含错误重试机制）
    
    Args:
        csv_file_path (str): CSV文件路径
        group_id (int): 组ID，默认12507
        max_retries (int): 最大重试次数，None表示无限重试
    
    Returns:
        float: F1分数，如果用户放弃返回None
    """
    
    url = "https://stat4011-part1.sta.cuhk.edu.hk/upload"
    attempt = 0
    
    while True:
        attempt += 1
        
        try:
            print(f"正在上传文件: {csv_file_path} (尝试 #{attempt})")
            
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
                    raise Exception("响应中未找到分数信息")
            else:
                raise Exception(f"上传失败，状态码: {response.status_code}, 响应: {response.text}")
                
        except FileNotFoundError:
            print(f"❌ 文件未找到: {csv_file_path}")
            return None
            
        except requests.exceptions.Timeout as e:
            error_msg = f"❌ 请求超时 (尝试 #{attempt}): {e}"
            print(error_msg)
            
        except requests.exceptions.ConnectionError as e:
            error_msg = f"❌ 连接错误 (尝试 #{attempt}): {e}"
            print(error_msg)
            
        except requests.exceptions.RequestException as e:
            error_msg = f"❌ 网络请求错误 (尝试 #{attempt}): {e}"
            print(error_msg)
            
        except Exception as e:
            error_msg = f"❌ 上传错误 (尝试 #{attempt}): {e}"
            print(error_msg)
        
        # 检查是否达到最大重试次数
        if max_retries is not None and attempt >= max_retries:
            print(f"❌ 达到最大重试次数 ({max_retries})，停止尝试")
            return None
        
        # 错误处理：提示用户并等待
        print(f"\n🚨 上传失败！可能原因：")
        print(f"   - 服务器暂时不可用")
        print(f"   - 网络连接问题")
        print(f"   - IP被临时限制")
        print(f"   - 服务器过载")
        
        user_input = input(f"\n服务器奔溃或ban请等待或尝试用新的IP\n输入 'retry' 重试，'quit' 退出，或直接Enter继续: ").strip().lower()
        
        if user_input == 'quit':
            print("❌ 用户选择退出上传")
            return None
        elif user_input == 'retry' or user_input == '':
            print(f"🔄 准备重试上传...")
            continue
        else:
            print(f"🔄 继续重试上传...")

def robust_upload_with_retry(csv_file_path, group_id=12507, auto_retry=True):
    """
    智能上传函数：自动重试 + 用户控制
    
    Args:
        csv_file_path (str): CSV文件路径
        group_id (int): 组ID
        auto_retry (bool): 是否自动重试，False时遇到错误直接失败
    
    Returns:
        float: F1分数
    """
    if auto_retry:
        return upload_file(csv_file_path, group_id, max_retries=None)  # 无限重试
    else:
        return upload_file(csv_file_path, group_id, max_retries=1)     # 只试一次

def upload_multiple_files(file_paths, group_id=12507):
    """
    批量上传多个文件
    
    Args:
        file_paths (list): 文件路径列表
        group_id (int): 组ID
    
    Returns:
        dict: {文件名: F1分数} 的字典
    """
    
    results = {}
    
    for file_path in file_paths:
        print(f"\n--- 上传文件 {file_path} ---")
        f1_score = upload_file(file_path, group_id)
        
        if f1_score is not None:
            results[file_path] = f1_score
        else:
            results[file_path] = None
        

    
    return results

def validate_csv_format(csv_file_path):
    """
    验证CSV文件格式是否正确
    
    Args:
        csv_file_path (str): CSV文件路径
    
    Returns:
        bool: 格式是否正确
    """
    
    try:
        import pandas as pd
        
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
    test_file = "/Users/mannormal/4011/Qi Zihan/v3/v0.1ensemble.csv"
    
    # 验证格式
    if validate_csv_format(test_file):
        # 上传文件
        result = upload_file(test_file)
        print(f"最终结果: {result}")
    else:
        print("文件格式验证失败，跳过上传")