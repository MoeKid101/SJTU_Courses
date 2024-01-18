import jittor as jt
from jittor import nn, Module
import matplotlib.pyplot as plt

class SimModel(Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.Relu()
        self.linear1 = nn.Linear(1, 64)
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, 1)

    def execute(self, x:jt.Var):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x

def fun_target(x:jt.Var, noise_std:jt.float32=100)->jt.Var:
    noise = jt.randn_like(x) * noise_std
    result = x*x*x + noise
    return result

def gen_data(low:jt.float32, high:jt.float32, num:int, function, noise_var:jt.float32):
    input = jt.rand([num, 1], dtype=jt.float32, requires_grad=False)
    input = (high-low)*input + low
    output = function(input, noise_var)
    return input, output

def train(model:SimModel, data:tuple, loss_function, optimizer:nn.Optimizer, max_itr:int):
    jt.flags.use_cuda = True
    model.train()
    train_losses = []
    inputs, targets = data
    for itr in range(max_itr):
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        optimizer.step(loss)
        train_losses.append(loss.data[0])
        if itr % 20 == 0:
            print(f'iteration times: {itr}, loss={loss.data[0]}')
            jt.save(model.state_dict(), 'Task1/model')
    plt.plot([i for i in range(len(train_losses))], train_losses, linewidth=1)
    plt.savefig('Task1/loss_stat.jpg')
    plt.clf()

def test(model:SimModel, data:tuple, loss_function, trainData:tuple):
    model.eval()
    inputs, answers = data
    inputs:jt.Var
    answers:jt.Var
    outputs = model(inputs)
    loss = loss_function(outputs, answers)
    # visualization
    print(f'loss={loss}')
    new_x = jt.linspace(-3, 3, 300).reshape([-1, 1])
    new_ans = fun_target(new_x, 0)
    new_prd = model(new_x)
    x_line, ans_line, prd_line = new_x.numpy(), new_ans.numpy(), new_prd.numpy()
    plt.scatter(trainData[0], trainData[1], s=2, c='red', alpha=0.3)
    plt.plot(x_line, ans_line, linewidth=3)
    plt.plot(x_line, prd_line, linewidth=3)
    plt.legend(['Train data', 'Test data', 'Prediction'])
    plt.ylim((-40, 40))
    plt.title('$\sigma=20$')
    plt.savefig('Task1/test_result.jpg', dpi=600)
    plt.clf()

def main():
    model = SimModel()
    # Training Parameters
    learning_rate, weight_decay, max_iteration = 1e-3, 1e-4, 1000
    loss_function = nn.MSELoss(reduction='mean')
    optimizer = nn.SGD(model.parameters(), learning_rate, weight_decay)
    # data generation
    train_data = gen_data(-3, 3, 800, fun_target, 20)
    test_data = gen_data(-3, 3, 200, fun_target, 0)
    # train
    train(model, train_data, loss_function, optimizer, max_iteration)
    # test
    test(model, test_data, loss_function, train_data)

if __name__ == '__main__':
    main()