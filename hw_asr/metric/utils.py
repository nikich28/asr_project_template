# Don't forget to support cases when target_text == ''
from editdistance import distance


def calc_cer(target_text, predicted_text) -> float:
    # TODO: your code here
    return distance(target_text, predicted_text) / len(target_text) if len(target_text) != 0 else len(predicted_text)


def calc_wer(target_text, predicted_text) -> float:
    # TODO: your code here
    preds = predicted_text.split(" ")
    targets = target_text.split(" ")
    return len(preds) if len(targets) == 0 else distance(targets, preds) / len(targets)
