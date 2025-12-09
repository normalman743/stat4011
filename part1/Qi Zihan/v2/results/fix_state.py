import json
import os

# 修复状态文件
state_file = "three_tier_state.json"

if os.path.exists(state_file):
    with open(state_file, 'r') as f:
        state = json.load(f)
    
    print("修复前的状态:")
    print(f"  反向搜索状态: {state.get('reverse_search_status', 'none')}")
    print(f"  反向队列长度: {len(state.get('reverse_queue', []))}")
    print(f"  确认bad: {len(state.get('confirmed_bad_ids', []))}")
    print(f"  确认good: {len(state.get('confirmed_good_ids', []))}")
    
    # 重置反向搜索状态
    state['reverse_search_status'] = 'pending'
    state['reverse_queue'] = []
    
    # 保存修复后的状态
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)
    
    print("\n修复后的状态:")
    print(f"  反向搜索状态: {state['reverse_search_status']}")
    print(f"  反向队列长度: {len(state['reverse_queue'])}")
    print("✅ 状态文件已修复，现在可以正常运行三类分层算法")
else:
    print("❌ 状态文件不存在")