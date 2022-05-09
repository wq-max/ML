class Perceptron(object):
    def __init__(self, input_para_num, acti_func):
        self.activator = acti_func
        self.weights = [0.0 for _ in range(input_para_num)]

    # 对象的“非正式”字符串表示形式
    def __str__(self):
        return 'final weights\n\tw0 = {:.2f}\n\tw1 = {:.2f}\n\tw2 = {:.2f}' \
            .format(self.weights[0], self.weights[1], self.weights[2])

    def predict(self, row_vec):
        act_values = 0.0
        for i in range(len(self.weights)):
            act_values += self.weights[i] * row_vec[i]
        return self.activator(act_values)

    # 训练权值
    def train(self, dataset, iteration, rate):
        for i in range(iteration):
            for input_vec_label in dataset:
                prediction = self.predict(input_vec_label)
                self._update_weights(input_vec_label, prediction, rate)

    def _update_weights(self, input_vec_label, prediction, rate):
        delta = input_vec_label[-1] - prediction
        for i in range(len(self.weights)):
            self.weights[i] += rate * delta * input_vec_label[i]


def get_training_dataset():
    dataset = [[-1, 1, 1, 1], [-1, 0, 0, 0], [-1, 1, 0, 0], [-1, 0, 1, 0]]
    return dataset


# 定义激活函数
def func_activator(input_value):
    return 1.0 if input_value >= 0.0 else 0.0


def train_and_perceptron():
    p = Perceptron(3, func_activator)
    dataset = get_training_dataset()
    p.train(dataset, 10, 0.1)
    return p


if __name__ == '__main__':
    and_perceptron = train_and_perceptron()
    print(and_perceptron)
    print('1 and 1 = %d' % and_perceptron.predict([-1, 1, 1]))
    print('0 and 0 = %d' % and_perceptron.predict([-1, 0, 0]))
    print('1 and 0 = %d' % and_perceptron.predict([-1, 1, 0]))
    print('0 and 1 = %d' % and_perceptron.predict([-1, 0, 1]))
