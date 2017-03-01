__author__ = 'Hang Wu'

from net import *

NUM_ROUNDS = 5
pars = {'batch_size': 100, 'std': 0,
        'lr': .1, 'momentum': .5, 'weight_decay':.0,
        'label_noise': (None, None),
        'has_dropout':1, 'has_bn':0,
        'use_cuda':1
        }
pars['cuda'] = 'use_cuda' in pars and pars['use_cuda'] and torch.cuda.is_available()
pars['train_loader'], pars['test_loader'] = get_loader(**pars)


res_dict ={
    "std":[],
    "scheme":[],
    "test_loss":[],
    "test_err":[],
    "rounds":[]
}

SCHEMES = {'Naive':(0,0,0), 'BatchNorm':(1,0,0), 'DropOut':(0,1,0), 'WeightDecay':(0,0,.0005), 'BN + DO':(1,1,0), 'BN + DO + WD':(1,1,.0005)}

#Testing the effect of noise on classification error
res = pd.DataFrame(res_dict)
for i in range(NUM_ROUNDS):
     print(i)
     for std in [0, 4, 8, 16, 32, 64, 128, 256, 512]:
         pars['std'] = std
         print("****************", std)
         for scheme in SCHEMES:
             print(scheme)
             pars['has_bn'], pars['has_dropout'], pars['weight_decay'] = SCHEMES[scheme]
             model = Net(has_dropout= pars['has_dropout'], has_bn=pars['has_bn'])
             t_loss, t_err, _,_,_ = eval_net(model, 1, **pars)
             temp = [ i, scheme, std, np.mean(t_err), t_loss]
             res.loc[len(res)] = temp

print(res)
res.to_pickle('out/exp2_1.dat')

#Testing which class is most sensitive to image noise

res2_dict = {'rounds':[], 'scheme':[], 'std':[], 'test_err':[], 'test_loss':[], 'class':[]}

res2 = pd.DataFrame(res2_dict)

pars['has_bn'], pars['has_dropout'], pars['weight_decay'] = (1,1,0)
for i in range(NUM_ROUNDS):
    print(i)
    for std in [0, 4, 8, 16, 32, 64, 128, 256, 512]:
        pars['std'] = std
        print("****************", std)
        for scheme in {'Naive':(0,0,0),'BN + DO':(1,1,0)}:
            print(scheme)
            pars['has_bn'], pars['has_dropout'], pars['weight_decay'] = SCHEMES[scheme]
            model = Net(has_dropout=pars['has_dropout'], has_bn=pars['has_bn'])
            t_loss, t_err, _, _, _ = eval_net(model, 1, **pars)
            for j in range(10):
                res2.loc[len(res2)] = [j, i, scheme, std, t_err[j], t_loss]

print(res2)
res2.to_pickle('out/exp2_2.dat')
