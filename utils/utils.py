import math

def cosine_learning_rate(current_round, total_rounds, initial_lr=0.001, min_lr=0):
    """
    Compute the learning rate based on a cosine schedule.

    :param current_round: The current training round (0-indexed).
    :param total_rounds: The total number of training rounds.
    :param initial_lr: The initial learning rate.
    :param min_lr: The minimum learning rate.
    :return: The computed learning rate for the current round.
    """
    # Compute the cosine learning rate
    cosine_lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * current_round / total_rounds))
    return cosine_lr


from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def calculate_matching_accuracy(labels, preds):
    # 提取最后的选项
    labels_option = labels[-1]
    preds_option = preds[-1]

    # 计算 precision、recall、f1 和 support
    precision, recall, f1, _ = precision_recall_fscore_support([labels_option], [preds_option], average='binary')

    # 计算准确率
    acc = accuracy_score([labels_option], [preds_option])

    return precision, recall, f1, acc



if __name__ == "__main__":

    # Example usage:
    num_rounds = 300
    initial_lr = 5e-5
    min_lr = 1e-6

    lr_list = []
    for round in range(num_rounds):
        lr = cosine_learning_rate(round, num_rounds, initial_lr, min_lr)
        lr_list.append(lr)
        print(f"Round {round + 1}/{num_rounds}, Learning Rate: {lr:.8f}")
