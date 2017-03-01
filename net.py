#Adapted from https://github.com/pytorch/examples/blob/master/mnist/main.py

from utils import *

def eval_net(model, n_epochs = 1,**kwargs):

    train_loader, test_loader = kwargs['train_loader'], kwargs['test_loader']
    optimizer = optim.SGD(model.parameters(), lr=kwargs['lr'], momentum=kwargs['momentum'], weight_decay=kwargs['weight_decay'])

    #for tracking the convergence w.r.t to mini-batches
    train_losses = []
    test_losses = []

    def train(epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if 'cuda' in kwargs and kwargs['cuda']:
                #GPU Training
                data, target = data.cuda(), target.cuda()
            if 'std' in kwargs:
                #Adding noises to images
                data = add_gaussian_noise(data, 0, kwargs['std']/255)
            if 'label_noise' in kwargs and kwargs['label_noise'][0] == 'permute':
                #Adding permutation label noise
                target = permute_label(target, kwargs['label_noise'][1])
            if 'label_noise' in kwargs and kwargs['label_noise'][0] == 'corrupt':
                #Adding random corruption noise
                target = random_corrupt_label(target, kwargs['label_noise'][1])

            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)

            train_losses.append(loss.data[0])

            loss.backward()
            optimizer.step()

    def test(epoch):
        model.eval()
        test_loss = 0
        correct = 0

        all_targets = []
        all_preds = []
        for data, target in test_loader:
            if 'cuda' in kwargs and kwargs['cuda']:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            test_loss += F.nll_loss(output, target).data[0]

            test_losses.append(F.nll_loss(output, target).data[0])

            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()
            all_targets += target.data.numpy().ravel().tolist()
            all_preds += pred.numpy().ravel().tolist()

        C, test_err = class_report(all_targets, all_preds)

        test_loss = test_loss
        test_loss /= len(test_loader)  # loss function already averages over batch size


        return test_loss, test_err, C

    for epoch in range(n_epochs):
        train(epoch)
        test_loss, test_err, C = test(epoch)
        print(np.mean(test_err))

    return test_loss, test_err, C, train_losses, test_losses

def get_loader(batch_size, **kwargs):
    loader_kwargs = {'num_workers': 1, 'pin_memory': True} if ('cuda' in kwargs and kwargs['cuda']) else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=batch_size, shuffle=True, **loader_kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor()
        ])),
        batch_size=batch_size, shuffle=True, **loader_kwargs)

    return train_loader, test_loader

#The neural net structure
class Net(nn.Module):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        if 'has_dropout' in kwargs and kwargs['has_dropout']:
            self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        if 'has_bn' in kwargs and kwargs['has_bn']:
            self.bn = nn.BatchNorm1d(320)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        if 'has_dropout' in self.kwargs and self.kwargs['has_dropout']:
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        else:
            x = F.relu(F.max_pool2d(self.conv2(x),2))
        x = x.view(-1, 320)
        if 'has_bn' in self.kwargs and self.kwargs['has_bn']:
            x = self.bn(x)
        x = F.relu(self.fc1(x))

        if 'has_droput' in self.kwargs and self.kwargs['has_dropout']:
            x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        return F.log_softmax(x)

if __name__ == '__main__':
    m = Net( has_droput=1)
