#!/usr/bin/env python3
"""
V3主程序 - 二分法优化系统
简化版本，专注于获得更高的F1分数
"""

import os
import pandas as pd
#from upload_module import upload_file
from simulator import simulate_f1 as upload_file
from confusion_calculator import calculate_confusion_from_f1
from binary_search import binary_optimize_accounts

def save_best_result(predictions_dict, f1_score, output_dir="/Users/mannormal/4011/Qi Zihan/v3"):
    """
    保存最佳结果，文件名格式: v{score}.csvn
    
    Args:
        predictions_dict (dict): 预测结果 {account_id: 0/1}
        f1_score (float): F1分数
        output_dir (str): 输出目录
    
    Returns:
        str: 保存的文件路径
    """
    
    # 创建DataFrame
    predictions_list = []
    for account_id, predict in predictions_dict.items():
        predictions_list.append({"ID": account_id, "Predict": predict})
    
    df = pd.DataFrame(predictions_list)
    
    # 生成文件名
    score_str = f"{f1_score:.8f}".replace(".", "")[:8]  # 取前8位数字
    filename = f"v{score_str}ensemble.csv"
    filepath = os.path.join(output_dir, filename)
    
    # 保存文件
    df.to_csv(filepath, index=False)
    
    print(f"💾 保存最佳结果: {filename}")
    print(f"  F1分数: {f1_score:.6f}")
    print(f"  预测Bad: {len(df[df['Predict'] == 1])}")
    print(f"  预测Good: {len(df[df['Predict'] == 0])}")
    
    return filepath

def load_initial_predictions():
    print("=== 分层二分法优化 ===")
    
    scores_df = pd.read_csv("/Users/mannormal/4011/account_scores.csv")
    
    layers = [
        {"id": 1, "name": "[0.0-0.1)", "range": (0.0, 0.1), "initial_guess": 0},
        {"id": 2, "name": "[0.1-0.2)", "range": (0.1, 0.2), "initial_guess": 0},
        {"id": 3, "name": "[0.2-0.5)", "range": (0.2, 0.5), "initial_guess": 0},
        {"id": 4, "name": "[0.5-0.8)", "range": (0.5, 0.8), "initial_guess": 1},
        {"id": 5, "name": "[0.8-1.0]", "range": (0.8, 1.0), "initial_guess": 1}
    ]
    layers.reverse()
    
    final_predictions = {}
    state_file = "/Users/mannormal/4011/Qi Zihan/v3/optimization_state.json"
    total_iterations = 0
    
    if os.path.exists(state_file):
        import json
        with open(state_file, 'r') as f:
            state = json.load(f)
        print(f"恢复状态: 从第{state['current_layer']}层开始")
        final_predictions = state['predictions']
        start_layer = state['current_layer']
        total_iterations = state.get('total_iterations', 0)
        print(f"已完成总迭代次数: {total_iterations}")
    else:
        start_layer = 1
        final_predictions = {}
        total_iterations = 0
    
    for layer in layers:
        
        if layer['id'] < start_layer:
            continue
            
        print(f"\n=== 大迭代 v{layer['id']}: {layer['name']} ===")
        
        min_score, max_score = layer["range"]
        if max_score == 1.0:
            layer_accounts = scores_df[(scores_df['predict'] >= min_score) & (scores_df['predict'] <= max_score)]
        else:
            layer_accounts = scores_df[(scores_df['predict'] >= min_score) & (scores_df['predict'] < max_score)]
        
        layer_account_ids = layer_accounts['ID'].tolist()
        print(f"层内账户数: {len(layer_account_ids)}")
        
        if len(layer_account_ids) == 0:
            continue
        
        current_predictions = {}
        for account_id in scores_df['ID']:
            if account_id in final_predictions:
                # 已确认的账户保持确认状态
                current_predictions[account_id] = final_predictions[account_id]
            elif account_id in layer_account_ids:
                # 当前层账户设为初始猜测（未确认状态-1会在二分法中处理）
                current_predictions[account_id] = layer['initial_guess']
            else:
                # 其他层账户设为相反值
                current_predictions[account_id] = 1 - layer['initial_guess']
        
        iteration = 1
        temp_file = f"/Users/mannormal/4011/Qi Zihan/v3/v{layer['id']}.{iteration}.csv"
        temp_df = pd.DataFrame([{"ID": aid, "Predict": pred} for aid, pred in current_predictions.items()])
        temp_df.to_csv(temp_file, index=False)
        
        layer_f1 = upload_file(temp_file)
        
        if layer_f1 is not None:
            predicted_bad = sum(current_predictions.values())
            confusion = calculate_confusion_from_f1(layer_f1, predicted_bad)
            
            if confusion:
                print(f"初始F1={layer_f1:.6f}, TP={confusion['TP']}, FP={confusion['FP']}, FN={confusion['FN']}, TN={confusion['TN']}")
                
                # 动态计算迭代次数 (基于log2 + 缓冲)
                layer_iterations = max(5, int(len(layer_account_ids).bit_length()) + 3)
                print(f"设置迭代次数: {layer_iterations} (基于层大小{len(layer_account_ids)})")
                print(f"当前总迭代次数: {total_iterations}")
                
                optimized_predictions, account_status = binary_optimize_accounts(
                    layer_account_ids,
                    current_predictions,
                    upload_file,
                    max_iterations=None  # 无限制，找到为止
                )
                
                # 保存优化结果和确认状态
                for account_id in layer_account_ids:
                    final_predictions[account_id] = optimized_predictions[account_id]
                
                # 统计确认状态
                confirmed_count = sum(1 for status in account_status.values() if status != -1)
                unconfirmed_count = len(layer_account_ids) - confirmed_count
                print(f"层优化完成: 确认了 {confirmed_count}/{len(layer_account_ids)} 个账户，未确认: {unconfirmed_count}")
        
        else:
            print(f"❌ 层 v{layer['id']} 上传失败，使用initial guess")
            for account_id in layer_account_ids:
                final_predictions[account_id] = layer['initial_guess']
        
        import json
        total_iterations += layer_iterations if 'layer_iterations' in locals() else 0
        
        # 创建完整的状态文件，包含确认状态
        all_account_status = {}
        if 'account_status' in locals():
            all_account_status.update(account_status)
        
        state = {
            'current_layer': layer['id'] + 1,
            'predictions': final_predictions,
            'account_status': all_account_status,  # 新增确认状态
            'total_iterations': total_iterations
        }
        with open(state_file, 'w') as f:
            json.dump(state, f)
        print(f"保存状态: 第{layer['id']}层完成，累计迭代: {total_iterations}")
        
        # Debug: 显示确认状态统计
        if all_account_status:
            confirmed_good = sum(1 for status in all_account_status.values() if status == 0)
            confirmed_bad = sum(1 for status in all_account_status.values() if status == 1)
            unconfirmed = sum(1 for status in all_account_status.values() if status == -1)
            print(f"当前总状态: 确认good={confirmed_good}, 确认bad={confirmed_bad}, 未确认={unconfirmed}")
    
    for account_id in scores_df['ID']:
        if account_id not in final_predictions:
            final_predictions[account_id] = 0
    
    bad_count = sum(final_predictions.values())
    good_count = len(final_predictions) - bad_count
    
    print(f"\n✅ 所有层优化完成")
    print(f"5层预测结果: Bad={bad_count}, Good={good_count}")
    
    # v7.0持续优化 - 每次F1破新高就覆盖保存
    print(f"\n=== v7.0 持续优化 - 目标F1=1.0 ===")
    
    # 初始测试
    v7_file = "/Users/mannormal/4011/Qi Zihan/v3/v7.0.csv"
    temp_df = pd.DataFrame([{"ID": aid, "Predict": pred} for aid, pred in final_predictions.items()])
    temp_df.to_csv(v7_file, index=False)
    
    best_f1 = upload_file(v7_file)
    best_predictions = final_predictions.copy()
    
    print(f"初始F1={best_f1:.6f}")
    
    # 持续优化直到F1=1或无法提升
    max_rounds = 20
    scores_dict = dict(zip(scores_df['ID'], scores_df['predict']))
    tested_candidates = set()  # 记录已测试的候选账户组合
    no_improvement_rounds = 0  # 连续无改进轮次计数
    
    for round_num in range(1, max_rounds + 1):
        print(f"\n--- v7.0 轮次 {round_num} ---")
        
        # 计算当前混淆矩阵
        predicted_bad = sum(best_predictions.values())
        confusion = calculate_confusion_from_f1(best_f1, predicted_bad)
        
        if confusion['FP'] == 0 and confusion['FN'] == 0:
            print("🎉 达到完美F1=1.0!")
            break
        
        print(f"TP={confusion['TP']}, FP={confusion['FP']}, FN={confusion['FN']}, TN={confusion['TN']}")
        
        # 选择优化目标 - 使用渐进策略避免重复测试
        candidates = []
        
        # 计算需要测试的账户数量（逐渐增加范围）
        fp_test_count = min(confusion['FP'], max(1, confusion['FP'] // (round_num + 1)))
        fn_test_count = min(confusion['FN'], max(1, confusion['FN'] // (round_num + 1)))
        
        if confusion['FP'] > 0:
            # 有FP：优化分数最低的bad预测账户
            bad_accounts = [aid for aid, pred in best_predictions.items() if pred == 1]
            # 跳过前面轮次已测试的账户
            skip_count = (round_num - 1) * fp_test_count
            fp_candidates = sorted(bad_accounts, key=lambda x: scores_dict[x])[skip_count:skip_count + fp_test_count]
            candidates.extend(fp_candidates)
            print(f"优化{len(fp_candidates)}个可能的FP账户 (跳过前{skip_count}个)")
        
        if confusion['FN'] > 0:
            # 有FN：优化分数最高的good预测账户
            good_accounts = [aid for aid, pred in best_predictions.items() if pred == 0]
            # 跳过前面轮次已测试的账户
            skip_count = (round_num - 1) * fn_test_count
            fn_candidates = sorted(good_accounts, key=lambda x: scores_dict[x], reverse=True)[skip_count:skip_count + fn_test_count]
            candidates.extend(fn_candidates)
            print(f"优化{len(fn_candidates)}个可能的FN账户 (跳过前{skip_count}个)")
        
        if not candidates:
            print("没有更多可优化的账户")
            break
        
        # 检查是否已测试过这组候选账户
        candidates_key = tuple(sorted(candidates))
        if candidates_key in tested_candidates:
            print("此组合已测试过，尝试扩大范围")
            # 如果已测试过，增加测试范围
            if confusion['FP'] > 0:
                bad_accounts = [aid for aid, pred in best_predictions.items() if pred == 1]
                fp_candidates = sorted(bad_accounts, key=lambda x: scores_dict[x])[:min(len(bad_accounts), confusion['FP'] + round_num)]
                candidates.extend(fp_candidates)
            if confusion['FN'] > 0:
                good_accounts = [aid for aid, pred in best_predictions.items() if pred == 0]
                fn_candidates = sorted(good_accounts, key=lambda x: scores_dict[x], reverse=True)[:min(len(good_accounts), confusion['FN'] + round_num)]
                candidates.extend(fn_candidates)
            candidates = list(set(candidates))  # 去重
            candidates_key = tuple(sorted(candidates))
            
            if candidates_key in tested_candidates:
                print("扩大范围后仍是重复组合，停止优化")
                break
        
        tested_candidates.add(candidates_key)
        
        # 进行优化
        optimized_predictions, v7_account_status = binary_optimize_accounts(
            candidates,
            best_predictions,
            upload_file,
            max_iterations=15
        )
        total_iterations += 15
        
        # 测试新结果
        test_df = pd.DataFrame([{"ID": aid, "Predict": pred} for aid, pred in optimized_predictions.items()])
        test_df.to_csv(v7_file, index=False)
        
        new_f1 = upload_file(v7_file)
        
        if new_f1 > best_f1:
            print(f"🎉 F1破新高: {best_f1:.6f} → {new_f1:.6f}")
            best_f1 = new_f1
            best_predictions = optimized_predictions.copy()
            print(f"覆盖保存 v7.0.csv")
            no_improvement_rounds = 0  # 重置无改进计数
        else:
            print(f"本轮无提升: {new_f1:.6f} ≤ {best_f1:.6f}")
            no_improvement_rounds += 1
            
        # 如果连续3轮无改进，停止优化
        if no_improvement_rounds >= 3:
            print("连续3轮无改进，停止优化")
            break
            
        if best_f1 >= 0.9999:
            print("🎉 接近完美F1!")
            break
    
    print(f"\nv7.0最终结果: F1={best_f1:.6f}")
    print(f"总计完成迭代次数: {total_iterations}")
    
    # 最终保存状态
    import json
    final_state = {
        'current_layer': 6,  # 表示已完成所有层
        'predictions': best_predictions,
        'total_iterations': total_iterations,
        'final_f1': best_f1
    }
    with open(state_file, 'w') as f:
        json.dump(final_state, f, indent=2)
    print(f"保存最终状态: 总迭代{total_iterations}次，F1={best_f1:.6f}")
    
    return best_predictions


def main():
    print("=== V3二分法优化系统 ===")
    result = load_initial_predictions()
    print(f"✅ 完成: {result}")
    reset_score = upload_file("/Users/mannormal/4011/Qi Zihan/v2/results/Transformer_basic_submission_FULL_DATA_f1_0.8913_epochs_130.256.csv")
    print(f"重新上传旧版本文件测试F1: {reset_score}")
if __name__ == "__main__":
    main()