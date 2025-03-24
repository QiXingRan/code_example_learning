import numpy as np

def softmax(input):
    max_num = np.max(input)
    # 防止指数上溢
    exp_sum = np.exp(input - max_num)
    probs = exp_sum / np.sum(exp_sum)
    return probs

input = [5, 6, 21]
print(softmax(input))