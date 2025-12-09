import json

# Debug反向队列问题
state_file = "three_tier_state.json"

with open(state_file, 'r') as f:
    state = json.load(f)

print("=== Debug反向队列问题 ===")
print(f"反向队列长度: {len(state.get('reverse_queue', []))}")
print(f"确认good数量: {len(state.get('confirmed_good_ids', []))}")

if len(state.get('reverse_queue', [])) > 0:
    print(f"\n前5个反向队列账户:")
    for i, acc in enumerate(state['reverse_queue'][:5]):
        print(f"  {i+1}. ID: {acc['ID']}, Score: {acc['score']}")

print(f"\n前5个确认good的ID:")
for i, acc_id in enumerate(state['confirmed_good_ids'][:5]):
    print(f"  {i+1}. {acc_id}")

# 检查队列中是否有已确认为good的账户
if len(state.get('reverse_queue', [])) > 0:
    queue_ids = set(acc['ID'] for acc in state['reverse_queue'])
    confirmed_good_ids = set(state['confirmed_good_ids'])
    
    overlap = queue_ids & confirmed_good_ids
    print(f"\n队列与确认good的重叠数量: {len(overlap)}")
    
    if len(overlap) > 0:
        print("前5个重叠ID:")
        for i, acc_id in enumerate(list(overlap)[:5]):
            print(f"  {i+1}. {acc_id}")

# 模拟队列更新
print(f"\n=== 模拟队列更新测试 ===")
if len(state.get('reverse_queue', [])) > 0:
    test_batch = state['reverse_queue'][:3]  # 取前3个测试
    test_ids = [acc['ID'] for acc in test_batch]
    
    print(f"测试批次ID: {test_ids}")
    
    # 模拟移除逻辑
    original_length = len(state['reverse_queue'])
    new_queue = [acc for acc in state['reverse_queue'] if acc['ID'] not in test_ids]
    new_length = len(new_queue)
    
    print(f"原始队列长度: {original_length}")
    print(f"移除后队列长度: {new_length}")
    print(f"应该移除数量: {len(test_ids)}")
    print(f"实际移除数量: {original_length - new_length}")
    
    if original_length - new_length == 0:
        print("❌ 队列没有被更新！")
        # 检查ID匹配问题
        print("检查ID匹配:")
        for test_id in test_ids:
            found = any(acc['ID'] == test_id for acc in state['reverse_queue'])
            print(f"  {test_id}: {'找到' if found else '未找到'}")
    else:
        print("✅ 队列更新正常")