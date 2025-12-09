import json

# 修复重复ID问题
state_file = "three_tier_state.json"

with open(state_file, 'r') as f:
    state = json.load(f)

print("修复前:")
print(f"  确认bad: {len(state.get('confirmed_bad_ids', []))}")
print(f"  确认good: {len(state.get('confirmed_good_ids', []))}")
print(f"  反向队列: {len(state.get('reverse_queue', []))}")

# 去重
state['confirmed_bad_ids'] = list(set(state.get('confirmed_bad_ids', [])))
state['confirmed_good_ids'] = list(set(state.get('confirmed_good_ids', [])))

print("\n修复后:")
print(f"  确认bad: {len(state['confirmed_bad_ids'])}")
print(f"  确认good: {len(state['confirmed_good_ids'])}")

# 检查总数是否合理
total_confirmed = len(state['confirmed_bad_ids']) + len(state['confirmed_good_ids'])
print(f"  总确认: {total_confirmed}")

if total_confirmed <= 7558:
    print("✅ 数量合理，保存修复后的状态")
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)
else:
    print("❌ 数量仍然异常，需要进一步检查")

# 检查重叠
bad_ids = set(state['confirmed_bad_ids'])
good_ids = set(state['confirmed_good_ids'])
overlap = bad_ids & good_ids

if len(overlap) > 0:
    print(f"⚠️  发现 {len(overlap)} 个重叠ID，需要清理")
    # 从good中移除与bad重叠的ID
    state['confirmed_good_ids'] = [id for id in state['confirmed_good_ids'] if id not in bad_ids]
    print(f"清理后确认good: {len(state['confirmed_good_ids'])}")
    
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)
    print("✅ 重叠问题已修复")
else:
    print("✅ 没有发现重叠ID")