def calculate_confusion_matrix_bad_f1(bad_f1, predict_good, real_good, predict_bad, real_bad):
    TN = (bad_f1 * (predict_bad + real_bad)) / 2
    FN = predict_bad - TN
    FP = real_bad - TN
    TP = predict_good - FP
    return TP, FP, FN, TN

def calculate_bad_f1_from_matrix(TP, FP, FN, TN):
    
    precision_bad = TN / (TN + FN) if (TN + FN) > 0 else 0
    recall_bad = TN / (TN + FP) if (TN + FP) > 0 else 0
    
    if (precision_bad + recall_bad) == 0:
        return 0
    
    bad_f1 = 2 * precision_bad * recall_bad / (precision_bad + recall_bad)
    return bad_f1



# 层级配置
LAYER_CONFIG = {
    "high_good": {
        "real_good": 6626,
        "real_bad": 154,
        "total": 6780, 
        "predict": 0
    },

    "mid": {
        "real_good": 168, 
        "real_bad": 124,
        "total": 292,
        "predict": 0
    },
    "high_bad": {
        "real_good": 37,
        "real_bad": 449, 
        "total": 486,
        "predict": 1
    }
}

def check_layer_config_as_parent():
    # 测试parent 是 high_good + mid ，A是 mid
    # PTP的意思是 Parent中的 True Positive 也就是 预测为good且真实为good的数量
    # PFP的意思是 Parent中的 False Positive 也就是 预测为good但真实为bad的数量
    # PTN的意思是 Parent中的 True Negative 也就是 预测为bad且真实为bad的数量
    # PFN的意思是 Parent中的 False Negative 也就是 预测为bad但真实为good的数量
    # 让我们来看 parent 是对于前两层（high_good + mid）都预测为 good (predict=0) 后一层 (high_bad) 预测为 bad (predict=1)
    # 也就是说 我们预测的good有 6780 + 292 = 7072 个，预测的bad有 486 个
    # 真实的good有 6626 + 168 + 37 = 6831 个，真实的bad有 154 + 124 + 449 = 727 个
    # 计算 Parent 的 bad F1
    PTP = LAYER_CONFIG['high_good']['real_good'] + LAYER_CONFIG['mid']['real_good']  
    PFP = LAYER_CONFIG['high_good']['real_bad'] + LAYER_CONFIG['mid']['real_bad']  
    PTN = LAYER_CONFIG['high_bad']['real_bad'] 
    PFN = LAYER_CONFIG['high_bad']['real_good'] 
    parent_bad_f1 = calculate_bad_f1_from_matrix(PTP, PFP, PFN, PTN)
    #print("Parent 混淆矩阵:")
    #print(f"  PTP: {PTP}, PFP: {PFP}, PFN: {PFN}, PTN: {PTN}")
    #print(f"Parent bad F1: {parent_bad_f1:.4f}")
    return parent_bad_f1, PTP, PFP, PFN, PTN

def varify_calculate_confusion_matrix_bad_f1():
    parent_bad_f1, PTP, PFP, PFN, PTN = check_layer_config_as_parent()
    #验证 calculate_confusion_matrix_bad_f1
    real_good = LAYER_CONFIG['high_good']['real_good'] + LAYER_CONFIG['mid']['real_good'] + LAYER_CONFIG['high_bad']['real_good']
    real_bad = LAYER_CONFIG['high_good']['real_bad'] + LAYER_CONFIG['mid']['real_bad'] + LAYER_CONFIG['high_bad']['real_bad']
    parent_predict_good = LAYER_CONFIG['high_good']['total'] + LAYER_CONFIG['mid']['total']
    parent_predict_bad = LAYER_CONFIG['high_bad']['total'] 
    PTP, PFP, PFN, PTN = calculate_confusion_matrix_bad_f1(parent_bad_f1, parent_predict_good, real_good, parent_predict_bad, real_bad)
    #print(f"Calculated from bad F1 - PTP: {PTP:.0f}, PFP: {PFP:.0f}, PFN: {PFN:.0f}, PTN: {PTN:.0f}")

def analyze_inversion_effect_simple_1(PTP, PFP, PFN, PTN,A_bad_f1, A_predict_good, A_predict_bad, A_predict):
    """"
    分析反转影响的简化版本
    Args:
        PTP, PFP, PFN, PTN: Parent 的混淆矩阵 
        A_bad_f1: A 的 bad F1 分数
        A_predict_good: A 预测为 good 的数量
        A_predict_bad: A 预测为 bad 的数量
        A_predict: A 的预测类别 (0 或 1)
    Returns:
    {
        "direction": direction,
        better_type: f"{better:.0f}",
        worse_type: f"{worse:.0f}", 
        "ΔTP": f"{PTP - ATP:.0f}",
        "ΔFP": f"{PFP - AFP:.0f}",
        "ΔFN": f"{PFN - AFN:.0f}",
        "ΔTN": f"{PTN - ATN:.0f}"
    }
    """
    real_good = PTP + PFN
    real_bad = PFP + PTN
    ATP, AFP, AFN, ATN = calculate_confusion_matrix_bad_f1(
        A_bad_f1, A_predict_good, real_good, A_predict_bad, real_bad
    )
    
    better = PTP - ATP
    worse = PFP - AFP
    
    if A_predict == 1:
        better_type = "better_in_bad"   # 识别bad更好
        worse_type = "worse_in_good"    # good识别变差
        direction = "good -> bad"
    else:
        better_type = "better_in_good"  # 识别good更好
        worse_type = "worse_in_bad"     # bad识别变差
        direction = "bad -> good"

    return {
        "direction": direction,
        better_type: f"{better:.0f}",
        worse_type: f"{worse:.0f}",
        "ΔTP": f"{PTP - ATP:.0f}",
        "ΔFP": f"{PFP - AFP:.0f}",
        "ΔFN": f"{PFN - AFN:.0f}",
        "ΔTN": f"{PTN - ATN:.0f}"
    }

def analyze_inversion_effect_simple_0(parent_f1, parent_predict_good, parent_predict_bad, C_bad_f1, C_predict_good, C_predict_bad, C_predict):
    """"
    分析反转影响的简化版本
    Args:
        parent's f1 + predicted good + predicted bad(需要修改)
        C_bad_f1: Child 的 bad F1 分数
        C_predict_good: Child 预测为 good 的数量
        C_predict_bad: Child 预测为 bad 的数量
        C_predict: Child 的预测类别 (0 或 1)
    Returns:
    {
        "direction": direction,
        better_type: f"{better:.0f}",
        worse_type: f"{worse:.0f}", 
        "ΔTP": f"{PTP - ATP:.0f}",
        "ΔFP": f"{PFP - AFP:.0f}",
        "ΔFN": f"{PFN - AFN:.0f}",
        "ΔTN": f"{PTN - ATN:.0f}"
    }
    """
    real_bad = LAYER_CONFIG['high_good']['real_bad'] + LAYER_CONFIG['mid']['real_bad'] + LAYER_CONFIG['high_bad']['real_bad']
    real_good = LAYER_CONFIG['high_good']['real_good'] + LAYER_CONFIG['mid']['real_good'] + LAYER_CONFIG['high_bad']['real_good']
    PTP, PFP, PFN, PTN = calculate_confusion_matrix_bad_f1(parent_f1, parent_predict_good, real_good, parent_predict_bad, real_bad)
    ATP, AFP, AFN, ATN = calculate_confusion_matrix_bad_f1(
        C_bad_f1, C_predict_good, real_good, C_predict_bad, real_bad
    )
    
    better = PTP - ATP
    worse = PFP - AFP
    
    better = abs(better)
    worse = abs(worse)
    if C_predict == 1:
        better_type = "better_in_bad"   # 识别bad更好
        worse_type = "worse_in_good"    # good识别变差
        direction = "good -> bad"
    else:
        better_type = "better_in_good"  # 识别good更好
        worse_type = "worse_in_bad"     # bad识别变差
        direction = "bad -> good"

    return {
        "direction": direction,
        better_type: f"{better:.0f}",
        worse_type: f"{worse:.0f}",
        "ΔTP": f"{PTP - ATP:.0f}",
        "ΔFP": f"{PFP - AFP:.0f}",
        "ΔFN": f"{PFN - AFN:.0f}",
        "ΔTN": f"{PTN - ATN:.0f}"
    }



def test1():
    # 和test一样 但是使用并print analyze_inversion_effect_simple的结果 不需要太多的其他的print和mark
    parent_bad_f1, PTP, PFP, PFN, PTN = check_layer_config_as_parent()
    A_bad_f1 = get_test_f1()
    A_predict_good = LAYER_CONFIG['high_good']['total']
    A_predict_bad = LAYER_CONFIG['mid']['total'] + LAYER_CONFIG['high_bad']['total']
    print(analyze_inversion_effect_simple_1(PTP, PFP, PFN, PTN, A_bad_f1, A_predict_good, A_predict_bad, 1))

def get_test_f1():
    ATP = LAYER_CONFIG['high_good']['real_good']  # 6626 ✓
    AFP = LAYER_CONFIG['high_good']['real_bad']   # 154 ✓
    AFN = LAYER_CONFIG['mid']['real_good'] + LAYER_CONFIG['high_bad']['real_good'] # 168 + 37 = 205 ✓
    ATN = LAYER_CONFIG['mid']['real_bad'] + LAYER_CONFIG['high_bad']['real_bad']  # 124 + 449 = 573 ✓
    
    A_bad_f1 = calculate_bad_f1_from_matrix(ATP, AFP, AFN, ATN)
    return A_bad_f1

def test():
    varify_calculate_confusion_matrix_bad_f1()
    parent_bad_f1, PTP, PFP, PFN, PTN = check_layer_config_as_parent()
    print(f"Parent bad F1: {parent_bad_f1:.4f}")
    print(f"PTP: {PTP}, PFP: {PFP}, PFN: {PFN}, PTN: {PTN}")
    # 假设 A 是 mid 反转 其他不变 
    # 也就是 high_good 预测 good (predict=0) mid 预测 bad (predict=1) high_bad 预测 bad (predict=1)
    # 此时的预测： good 有 6780 个，预测 bad 有 292 + 486 = 778 个 
    # 在 good 中 真实 good 有 6626 个，真实 bad 有 154 个
    # 在 bad 中 真实 good 有 168 + 37 = 205 个，真实 bad 有 124 + 449 = 573 个
    # calculate A 的 bad F1 参数为： calculate_bad_f1_from_matrix(TP, FP, FN, TN):
    ATP = LAYER_CONFIG['high_good']['real_good']  # 6626 ✓
    AFP = LAYER_CONFIG['high_good']['real_bad']   # 154 ✓
    AFN = LAYER_CONFIG['mid']['real_good'] + LAYER_CONFIG['high_bad']['real_good'] # 168 + 37 = 205 ✓
    ATN = LAYER_CONFIG['mid']['real_bad'] + LAYER_CONFIG['high_bad']['real_bad']  # 124 + 449 = 573 ✓
    
    A_bad_f1 = calculate_bad_f1_from_matrix(ATP, AFP, AFN, ATN)
    print(f"A (mid 反转) bad F1: {A_bad_f1:.4f}")
    #varify
    real_good = LAYER_CONFIG['high_good']['real_good'] + LAYER_CONFIG['mid']['real_good'] + LAYER_CONFIG['high_bad']['real_good']
    real_bad = LAYER_CONFIG['high_good']['real_bad'] + LAYER_CONFIG['mid']['real_bad'] + LAYER_CONFIG['high_bad']['real_bad']
    A_predict_good = LAYER_CONFIG['high_good']['total']
    A_predict_bad = LAYER_CONFIG['mid']['total'] + LAYER_CONFIG['high_bad']['total']
    ATP2, AFP2, AFN2, ATN2 = calculate_confusion_matrix_bad_f1(A_bad_f1, A_predict_good, real_good, A_predict_bad, real_bad)
    assert abs(ATP - ATP2) < 1e-6
    assert abs(AFP - AFP2) < 1e-6
    assert abs(AFN - AFN2) < 1e-6
    assert abs(ATN - ATN2) < 1e-6
    print(f"Calculated from A bad F1 - ATP: {ATP2:.0f}, AFP: {AFP2:.0f}, AFN: {AFN2:.0f}, ATN: {ATN2:.0f}")
    print()
    # 分析反转影响
    # 先写一下 我们知道的情况 以及 我们希望验证的数值 PTP, PFP, PFN, PTN 是已知的
    # PTP的意思是 Parent中的 True Positive 也就是 预测为good且真实为good的数量 = 6794 （= 6626 + 168）
    # PFP的意思是 Parent中的 False Positive 也就是 预测为good但真实为bad的数量 = 278 = (154 + 124)
    # PTN的意思是 Parent中的 True Negative 也就是 预测为bad且真实为bad的数量 = 449 = (449)
    # PFN的意思是 Parent中的 False Negative 也就是 预测为bad但真实为good的数量 = 37 = (37)
    # 让我们来看 parent 是对于前两层（high_good + mid）都预测为 good (predict=0) 后一层 (high_bad) 预测为 bad (predict=1)
    # 也就是说 我们预测的good有 6780 + 292 = 7072 个，预测的bad有 486 个
    # 真实的good有 6626 + 168 + 37 = 6831 个，真实的bad有 154 + 124 + 449 = 727 个
    # 这个是parent
    # 下面是A（mid反转）中我们希望了解的数值 我们想知道mid中有多少 good 预测为 bad 多少 bad 预测为 bad 预计得到的结构是 mid real_good 168 mid real bad 124 我们目前知道 mid total 292
    # 下面是已知信息
    # A 有 6780 个 good 预测为 good
    # A 有 292 + 486 = 778 个 bad 预测为 bad
    # 在 good 中 真实 good 有 6626 个，真实 bad 有 154 个
    # 在 bad 中 真实 good 有 168 + 37 = 205 个，真实 bad 有 124 + 449 = 573 个
    # 也就是说 我们知道 A 的 TP, FP, FN, TN 分别是 6626, 154, 205, 573
    # 现在我们想知道 mid 中的 168 个 good 预测为 bad 的数量 以及 124 个 bad 预测为 bad 的数量
    # 以及 B 也就是 没有反转的 high_good 中 real good = 6626, real bad = 154
    DTP = PTP - ATP  # 168 A比parent多了 168 个 预测为good且真实为good的数量
    DFP = PFP - AFP  # 124 A比parent多了 124 个 预测为good但真实为bad的数量
    DFN = PFN - AFN  # -168 A比parent少了 168 个 预测为bad但真实为good的数量
    DTN = PTN - ATN  # -124 A比parent少了 124 个 预测为bad且真实为bad的数量
    print(f"D 混淆矩阵 (mid 反转部分):")
    print(f"  DTP: {DTP}, DFP: {DFP}, DFN: {DFN}, DTN: {DTN}")
    mid_predict_good = DTP 
    mid_predict_bad = DFN
    assert mid_predict_good == 168
    assert mid_predict_bad == 124
    # A 总共改变的数量是 168 + 124 = 292
    print(f"mid 预测为 good 的数量: {mid_predict_good}, 预测为 bad 的数量: {mid_predict_bad}")

def Init_parent_info():
    parent_bad_f1, PTP, PFP, PFN, PTN = check_layer_config_as_parent()
    parent_predict_good = LAYER_CONFIG['high_good']['total'] + LAYER_CONFIG['mid']['total']
    parent_predict_bad = LAYER_CONFIG['high_bad']['total'] 
    real_good = LAYER_CONFIG['high_good']['real_good'] + LAYER_CONFIG['mid']['real_good'] + LAYER_CONFIG['high_bad']['real_good']
    real_bad = LAYER_CONFIG['high_good']['real_bad'] + LAYER_CONFIG['mid']['real_bad'] + LAYER_CONFIG['high_bad']['real_bad']
    return parent_bad_f1, parent_predict_good, parent_predict_bad, real_good, real_bad

def get_A_info():
    A_bad_f1 = get_test_f1()
    A_predict_good = LAYER_CONFIG['high_good']['total']
    A_predict_bad = LAYER_CONFIG['mid']['total'] + LAYER_CONFIG['high_bad']['total']
    return A_bad_f1, A_predict_good, A_predict_bad

def main():
    print("Testing confusion matrix calculations with layer configurations...")
    #calculate_confusion_matrix_bad_f1(bad_f1, predict_good, real_good, predict_bad, real_bad):Parent bad F1: 0.7403 PTP: 6794, PFP: 278, PFN: 37, PTN: 449
    parent_bad_f1, parent_predict_good, parent_predict_bad, real_good, real_bad = Init_parent_info()
    # parent_bad_f1 parent_predict_good parent_predict_bad real_good real_bad is what we know
    PTP, PFP, PFN, PTN = calculate_confusion_matrix_bad_f1(parent_bad_f1, parent_predict_good, real_good, parent_predict_bad, real_bad)

    #二分法 eg mid 我们知道的是mid的 f1 score 以及 mid_predict_good  mid_predict_bad 
    A_bad_f1, A_predict_good, A_predict_bad = get_A_info()

    print(analyze_inversion_effect_simple_1(PTP, PFP, PFN, PTN, A_bad_f1, A_predict_good, A_predict_bad, 1))

if __name__ == "__main__":
    main()