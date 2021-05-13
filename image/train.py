import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image
from mixup_dataset import MixupInputDataset
from models import ImageClassifier
import fire

state = {}

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))

def train(net, train_loader_in, train_loader_out, optimizer, scheduler=None):
    net.train()  # enter train mode
    loss_avg = 0.0

    # start at a random point of the outlier dataset; this induces more randomness without obliterating locality
#     train_loader_out.dataset.offset = np.random.randint(len(train_loader_out.dataset))
    for in_set, out_set in tqdm(zip(train_loader_in, train_loader_out)):
        data = torch.cat((in_set[0], out_set[0]), 0)
        target = in_set[1]

        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        # forward
        x = net(data)

        # backward
        optimizer.zero_grad()

        loss = F.cross_entropy(x[:len(in_set[0])], target)
        # cross-entropy from softmax distribution to uniform distribution
        loss += 0.5 * -(x[len(in_set[0]):].mean(1) - torch.logsumexp(x[len(in_set[0]):], dim=1)).mean()

        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # exponential moving average
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2

    state['train_loss'] = loss_avg

# test function
def test(net, test_loader):
    net.eval()
    loss_avg = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            # forward
            output = net(data)
            loss = F.cross_entropy(output, target)

            # accuracy
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()

            # test loss average
            loss_avg += float(loss.data)

    state['test_loss'] = loss_avg / len(test_loader)
    state['test_accuracy'] = correct / len(test_loader.dataset)

def get_probs(net, loader):
    net.eval()
    scores = None
    with torch.no_grad():
        for data, target in tqdm(loader):
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            output = net(data)
            score = F.softmax(output, dim=1)
            score = score.detach().cpu().numpy()
            if scores is None:
                scores = score
            else:
                scores = np.concatenate((scores, score), axis=0)
    return np.array(scores)

def main(dataset='cifar10', model='resnet18', epochs=50, learning_rate=1e-1,
    batch_size=128, oe_batch_size=256, test_bs=256, momentum=0.9, decay=5e-4,
    save='./ckpts/oe_scratch', load='', ngpu=1, prefetch=4, lam=0.5, lam_direction='inter'):

    if not torch.cuda.is_available():
        ngpu = 0

    global state
    state = locals()
    print(state)
    torch.manual_seed(1)
    np.random.seed(1)

    # mean and standard deviation of channels of CIFAR-10 images
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                                trn.ToTensor(), trn.Normalize(mean, std)])
    test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

    if dataset == 'cifar10':
        train_data_in = dset.CIFAR10('./data', train=True, transform=train_transform, download=True)
        train_data_raw = dset.CIFAR10('./data', train=True)
        test_data = dset.CIFAR10('./data', train=False, transform=test_transform, download=True)
        num_classes = 10
    else:
        train_data_in = dset.CIFAR100('./data', train=True, transform=train_transform)
        train_data_raw = dset.CIFAR100('./data', train=True)
        test_data = dset.CIFAR100('./data', train=False, transform=test_transform)
        num_classes = 100

    calib_indicator = ''
    # if calibration:
    #     train_data_in, val_data = validation_split(train_data_in, val_share=0.1)
    #     calib_indicator = '_calib'

    ood_data = MixupInputDataset(train_data_raw, lam_mag=0.5, lam_random=False, lam_direction='inter', transform=train_transform)

    train_loader_in = torch.utils.data.DataLoader(
        train_data_in,
        batch_size=batch_size, shuffle=True,
        num_workers=prefetch, pin_memory=False)

    train_loader_out = torch.utils.data.DataLoader(
        ood_data,
        batch_size=oe_batch_size, shuffle=True, pin_memory=False)

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size, shuffle=False,
        num_workers=prefetch, pin_memory=False)

    # Create model
    try:
        net = ImageClassifier(num_classes=10, model=model)
    except Exception as e:
        print("Unsupported model, exeption:", e)
        return

    # Restore model
    model_found = False
    if load != '':
        for i in range(1000 - 1, -1, -1):
            model_name = os.path.join(load, dataset + calib_indicator + '_' + model +
                                    '_baseline_epoch_' + str(i) + '.pt')
            if os.path.isfile(model_name):
                net.load_state_dict(torch.load(model_name))
                print('Model restored! Epoch:', i)
                model_found = True
                break
        if not model_found:
            assert False, "could not find model to restore"

    if ngpu > 1:
        net = torch.nn.DataParallel(net, device_ids=list(range(ngpu)))

    if ngpu > 0:
        net.cuda()
        torch.cuda.manual_seed(1)

    cudnn.benchmark = True  # fire on all cylinders

    optimizer = torch.optim.SGD(
        net.parameters(), state['learning_rate'], momentum=state['momentum'],
        weight_decay=state['decay'], nesterov=True)


    # scheduler = torch.optim.lr_scheduler.LambdaLR(
    #     optimizer,
    #     lr_lambda=lambda step: cosine_annealing(
    #         step,
    #         epochs * len(train_loader_in),
    #         1,  # since lr_lambda computes multiplicative factor
    #         1e-6 / learning_rate))

    def set_lr(epoch):
        if epoch == 100:
            return 0.1
        elif epoch == 150:
            return 0.1
        else:
            return 1
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, set_lr, verbose=True)
    
    if not os.path.exists(save):
        os.makedirs(save)
    if not os.path.isdir(save):
        raise Exception('%s is not a dir' % save)

    with open(os.path.join(save, dataset + calib_indicator + '_' + model +
                                    '_oe_scratch_training_results.csv'), 'w') as f:
        f.write('epoch,time(s),train_loss,test_loss,test_error(%)\n')

    print('Beginning Training\n')

    # Main loop
    best_acc = 0.0
    for epoch in tqdm(range(0, epochs)):
        state['epoch'] = epoch

        begin_epoch = time.time()

        train(net, train_loader_in, train_loader_out, optimizer, scheduler)
        test(net, test_loader)

        if state['test_accuracy'] > best_acc:
            print(f"Found higher accuracy = {state['test_accuracy']}, saving best model...")
            torch.save(net.state_dict(),
                os.path.join(save, dataset + calib_indicator + '_' + model +
                                f'_oe_scratch_{lam}_{lam_direction}_best.pt'))
            best_acc = state['test_accuracy']

        # Save model
        torch.save(net.state_dict(),
                os.path.join(save, dataset + calib_indicator + '_' + model +
                                f'_oe_scratch_{lam}_{lam_direction}_epoch_' + str(epoch) + '.pt'))
        # Let us not waste space and delete the previous model
        prev_path = os.path.join(save, dataset + calib_indicator + '_' + model +
                                f'_oe_scratch_{lam}_{lam_direction}_epoch_' + str(epoch - 1) + '.pt')
        if os.path.exists(prev_path): os.remove(prev_path)

        # Show results

        with open(os.path.join(save, dataset + calib_indicator + '_' + model +
                                        f'_oe_scratch_{lam}_{lam_direction}_training_results.csv'), 'a') as f:
            f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' % (
                (epoch + 1),
                time.time() - begin_epoch,
                state['train_loss'],
                state['test_loss'],
                100 - 100. * state['test_accuracy'],
            ))

        # # print state with rounded decimals
        # print({k: round(v, 4) if isinstance(v, float) else v for k, v in state.items()})

        print('Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} | Test Error {4:.2f}'.format(
            (epoch + 1),
            int(time.time() - begin_epoch),
            state['train_loss'],
            state['test_loss'],
            100 - 100. * state['test_accuracy'])
        )

    # Evaluate best models
    best_path = os.path.join(save, dataset + calib_indicator + '_' + model +
                                f'_oe_scratch_{lam}_{lam_direction}_best.pt')
    checkpoint = torch.load(best_path)
    net.load_state_dict(checkpoint)

    ood_test_data = dset.CIFAR100('./data', download=True, transform=test_transform, train=False)
    ood_test_loader = torch.utils.data.DataLoader(
        ood_test_data,
        batch_size=batch_size, shuffle=False,
        num_workers=prefetch, pin_memory=False)
    
    ood_probs = get_probs(net, ood_test_loader)
    ood_msp = ood_probs.max(axis=1)

    id_probs = get_probs(net, test_loader)
    id_msp = id_probs.max(axis=1)

    if not os.path.exists('results'):
        os.makedirs('results')

    np.save(os.path.join('results', f'oe_{lam}_{lam_direction}_id_msp'), id_msp)
    np.save(os.path.join('results', f'oe_{lam}_{lam_direction}_ood_msp'), ood_msp)

if __name__ == '__main__':
    fire.Fire(main)
    # main(epochs=1)
