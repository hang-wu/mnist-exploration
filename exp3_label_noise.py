__author__ = 'Hang Wu'

from utils import *
from net import *
NUM_ROUNDS = 5
SCHEMES = {'Naive':(0,0,0), 'BatchNorm':(1,0,0), 'DropOut':(0,1,0), 'WeightDecay':(0,0,.0005), 'BN + DO':(1,1,0), 'BN + DO + WD':(1,1,.0005)}


pars = {'batch_size': 100,
        'lr': .1, 'momentum': .5,
        'label_noise': ['permute', 0]
        }
pars['train_loader'], pars['test_loader'] = get_loader(**pars)

res_dict = {'rounds':[], 'scheme':[], 'noise':[], 'test_loss':[], 'test_err':[]}

res = pd.DataFrame(res_dict)
for i in range(NUM_ROUNDS):
    print(i)
    for p in range(0,105,5):
        pars['label_noise'][1] = p/100
        print("****************", p/100)
        for scheme in {'Naive':(0,0,0),'BN + DO':(1,1,0)}:
            print(scheme)
            pars['has_bn'], pars['has_dropout'], pars['weight_decay'] = SCHEMES[scheme]
            model = Net(has_dropout=pars['has_dropout'], has_bn=pars['has_bn'])
            t_loss, t_err, _, _, _ = eval_net(model, 1, **pars)
            temp = [p/100, i, scheme, np.mean(t_err), t_loss]
            res.loc[len(res)] = temp

print(res)

res.to_pickle('out/exp3.dat')
