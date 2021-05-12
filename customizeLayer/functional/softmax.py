import torch
import numpy as np
def softmax_vector(vector: torch.Tensor):
    """输入为一维张量
    """
    # print(vector.shape)
    sigma = torch.zeros(1)
    # print(sigma)
    # print(vector)
    # for i in vector:
    #     vector[i] = vector[i] - max(vector)
    # print(vector)
    for i in vector:
        sigma = sigma + np.e ** i

    result = torch.zeros(vector.shape[0])
    for i in range(len(result)):
        result[i] = np.e ** vector[i] / sigma

    return result

def softmax_tensor(tensor: torch.Tensor):
    """输入可以是(*, *, *) 三维矩阵，对最后一维进行softmax
    """
    results = torch.zeros((tensor.shape[0], tensor.shape[1], tensor.shape[2]))
    # print(results.shape)
    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            results[i][j] = softmax_vector(tensor[i][j])


    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            for k in range(tensor.shape[2]):
                if torch.isnan(results[i][j][k]):
                    results[i][j][k] = 0
    return results


if __name__=="__main__":
    input = torch.randn((1, 12, 12))
    print(softmax_tensor(input))
