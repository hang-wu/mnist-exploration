__author__ = 'Hang Wu'

from net import *
NUM_ROUNDS = 5
import pickle

pars = {'batch_size': 100, 'std': 0,
        'lr': .1, 'momentum': .5, 'weight_decay':.0,
        'label_noise': (None, None),
        'has_dropout':1, 'has_bn':0
        }
pars['train_loader'], pars['test_loader'] = get_loader(**pars)
pars['has_bn'], pars['has_dropout'], pars['weight_decay'] = (1,1,0)

#for saving results for the rounds and later output the average
t_loss = 0.
t_err = np.zeros(10)
t_C = 0

all_train_losses = []

for i in range(NUM_ROUNDS):
    print("***********",i)
    model = Net(has_dropout= pars['has_dropout'], has_bn=pars['has_bn'])
    i_loss, i_err, i_C, train_losses, _ = eval_net(model, 1, **pars)
    t_err += np.array(i_err) / NUM_ROUNDS
    t_loss += i_loss / NUM_ROUNDS
    t_C += i_C
    all_train_losses.append(train_losses)

print(t_C)
C = t_C.astype(np.float32)

#Normalizing the confusion matrix to [0,1]
for i in range(10):
    s = sum(C[i,:])
    C[i,:] = C[i,:] / s

np.savetxt('out/conf_mat.csv', C, delimiter=',', fmt='%.3f')

print(t_loss)

print(t_err, np.mean(t_err))
np.savez('out/exp1_losses.npz', all_train_losses = np.array(all_train_losses))
