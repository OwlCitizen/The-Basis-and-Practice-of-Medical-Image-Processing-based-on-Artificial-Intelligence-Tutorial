#utils.py

def dice_score(pred, label):
    assert pred.shape == label.shape
    return 2*(pred*label).sum()/(pred.sum()+label.sum())