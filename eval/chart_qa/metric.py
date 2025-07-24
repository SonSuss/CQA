import difflib


def RelaxedAccuracy(pred, gt, threshold=0.05):
    try:
        gt = float(gt)
        pred = float(pred)
        if gt == 0.0:
            if pred == gt:
                return True
            else:
                return False
        else:
            if abs(pred-gt) / gt <= threshold:
                return True
            else:
                return False
    except:
        ratio = difflib.SequenceMatcher(None, str(pred), str(gt)).ratio()
        return True if ratio >= 1-threshold else False
        
def chartqa_evaluator(data, key='final_model_answer'):
    relaxed_acc = 0
    standard_acc = 0
    for item in data:
        item_relaxed_acc = RelaxedAccuracy(item[key], item['gt_answer'])
        item_standard_acc = True if str(item[key]) == str(item['gt_answer']) else False
        if item_relaxed_acc:
            relaxed_acc += 1
        if item_standard_acc:
            standard_acc += 1
    relaxed_accuracy = relaxed_acc/len(data)
    standard_accuracy = standard_acc/len(data)
    return standard_accuracy, relaxed_accuracy

