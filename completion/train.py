import torch.optim as optim
import torch
# from utils.train_utils import *
from train_utils import *
import logging
import math
import importlib
import datetime
import random
import munch
import yaml
import os
import sys
import argparse
from dataset import MVP_CP,ShapeNetH5
from models.CP3 import Model_step2

import warnings
warnings.filterwarnings("ignore")


def train():
    logging.info(str(args))
    if args.eval_emd:
        metrics = ['cd_p', 'cd_t', 'emd', 'f1']
    else:
        metrics = ['cd_p', 'cd_t', 'f1']
    best_epoch_losses = {m: (0, 0) if m == 'f1' else (0, math.inf) for m in metrics}
    train_loss_meter = AverageValueMeter()
    val_loss_meters = {m: AverageValueMeter() for m in metrics}

    # dataset = MVP_CP(prefix="train")
    # dataset_test = MVP_CP(prefix="val")

    dataset = ShapeNetH5(train=True, npoints=args.num_points_step2)
    dataset_test = ShapeNetH5(train=False, npoints=args.num_points_step2)



    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=int(args.workers))
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size,
                                            shuffle=False, num_workers=int(args.workers))
    logging.info('Length of train dataset:%d', len(dataset))
    logging.info('Length of test dataset:%d', len(dataset_test))

    if not args.manual_seed:
        seed = random.randint(1, 10000)
    else:
        seed = int(args.manual_seed)
    logging.info('Random Seed: %d' % seed)
    random.seed(seed)
    torch.manual_seed(seed)

    model_module = importlib.import_module('.%s' % args.model_name, 'models')
    net = torch.nn.DataParallel(model_module.Model(args))
    net.cuda()
    net2 = torch.nn.DataParallel(Model_step2(args))
    net2.cuda()

    if hasattr(model_module, 'weights_init'):
        net.module.apply(model_module.weights_init)
        net2.module.apply(model_module.weights_init)




    cascade_gan = (args.model_name == 'cascade')
    net_d = None
    if cascade_gan:
        net_d = torch.nn.DataParallel(model_module.Discriminator(args))
        net_d.cuda()
        net_d.module.apply(model_module.weights_init)

    lr = args.lr
    if cascade_gan:
        lr_d = lr / 2
    if args.lr_decay:
        if args.lr_decay_interval and args.lr_step_decay_epochs:
            raise ValueError('lr_decay_interval and lr_step_decay_epochs are mutually exclusive!')
        if args.lr_step_decay_epochs:
            decay_epoch_list = [int(ep.strip()) for ep in args.lr_step_decay_epochs.split(',')]
            decay_rate_list = [float(rt.strip()) for rt in args.lr_step_decay_rates.split(',')]

    optimizer = getattr(optim, args.optimizer)
    if args.optimizer == 'Adagrad':
        # optimizer = optimizer(net.module.parameters(), lr=lr, initial_accumulator_value=args.initial_accum_val)
        optimizer = optimizer(net2.module.parameters(), lr=lr, initial_accumulator_value=args.initial_accum_val)
    
    else:
        betas = args.betas.split(',')
        betas = (float(betas[0].strip()), float(betas[1].strip()))
        optimizer = optimizer(net2.module.parameters(), lr=lr, weight_decay=args.weight_decay, betas=betas)

    if cascade_gan:
        optimizer_d = optim.Adam(net_d.parameters(), lr=lr_d, weight_decay=0.00001, betas=(0.5, 0.999))

    alpha = None
    if args.varying_constant:
        varying_constant_epochs = [int(ep.strip()) for ep in args.varying_constant_epochs.split(',')]
        varying_constant = [float(c.strip()) for c in args.varying_constant.split(',')]
        assert len(varying_constant) == len(varying_constant_epochs) + 1

    if args.load_model:
        ckpt = torch.load(args.load_model)
        net.module.load_state_dict(ckpt['net_state_dict'])
        for p in net.parameters():
            p.requires_grad = False
        if args.load_model2:
            ckpt2 = torch.load(args.load_model)
            pretrained_dict = ckpt2['net_state_dict']
            model_dict = net2.module.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            net2.module.load_state_dict(model_dict)
            print('load model refinement')

        if cascade_gan:
            net_d.module.load_state_dict(ckpt['D_state_dict'])
        logging.info("%s's previous weights loaded." % args.model_name)

    for epoch in range(args.start_epoch, args.nepoch):
        #test

        train_loss_meter.reset()
        net.module.eval()
        net2.module.train()
        if cascade_gan:
            net_d.module.train()

        if args.varying_constant:
            for ind, ep in enumerate(varying_constant_epochs):
                if epoch < ep:
                    alpha = varying_constant[ind]
                    break
                elif ind == len(varying_constant_epochs)-1 and epoch >= ep:
                    alpha = varying_constant[ind+1]
                    break

        if args.lr_decay:
            if args.lr_decay_interval:
                if epoch > 0 and epoch % args.lr_decay_interval == 0:
                    lr = lr * args.lr_decay_rate
            elif args.lr_step_decay_epochs:
                if epoch in decay_epoch_list:
                    lr = lr * decay_rate_list[decay_epoch_list.index(epoch)]
            if args.lr_clip:
                lr = max(lr, args.lr_clip)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        for i, data in enumerate(dataloader, 0):
            optimizer.zero_grad()
            if cascade_gan:
                optimizer_d.zero_grad()

            _,lbs, inputs, gt = data
            # mean_feature = None

            inputs = inputs.float().cuda()
            gt = gt.float().cuda()
            lbs = lbs.float().cuda()
            inputs = inputs.transpose(2, 1).contiguous()
            result_dict= net(inputs, gt, alpha=alpha,prefix="val")
            final_out, loss_cd, total_train_loss = net2(lbs,result_dict['up_features'],result_dict['out2'],result_dict['global_feat'],gt, alpha=alpha)

            train_loss_meter.update(total_train_loss.mean().item())
            total_train_loss.backward(torch.squeeze(torch.ones(torch.cuda.device_count())).cuda())
            optimizer.step()

            if i % args.step_interval_to_print == 0:
                logging.info(exp_name + ' train [%d: %d/%d]  loss_type: %s, fine_loss: %f total_loss: %f lr: %f' %
                             (epoch, i, len(dataset) / args.batch_size, args.loss, loss_cd.mean().item(), total_train_loss.mean().item(), lr) + ' alpha: ' + str(alpha))

        if epoch % args.epoch_interval_to_save == 0:
            save_model('%s/network.pth' % log_dir, net, net_d=net_d)
            logging.info("Saving net...")

        if epoch % args.epoch_interval_to_val == 0 or epoch == args.nepoch - 1:
            val(net,net2, epoch, val_loss_meters, dataloader_test, best_epoch_losses)


def val(net,net2, curr_epoch_num, val_loss_meters, dataloader_test, best_epoch_losses):
    logging.info('Testing...')
    for v in val_loss_meters.values():
        v.reset()
    net.module.eval()
    net2.module.eval()

    with torch.no_grad():
        for i, data in enumerate(dataloader_test):
            _, lbs, inputs, gt = data
            # mean_feature = None
            curr_batch_size = gt.shape[0]

            inputs = inputs.float().cuda()
            gt = gt.float().cuda()
            lbs= lbs.float().cuda()
            inputs = inputs.transpose(2, 1).contiguous()

            
            result_dict = net(inputs, gt, prefix="val")
            result_dict2 = net2(lbs,result_dict['up_features'],result_dict['out2'],result_dict['global_feat'],gt,prefix="val")

            for k, v in val_loss_meters.items():
                v.update(result_dict2[k].mean().item(), curr_batch_size)

        fmt = 'best_%s: %f [epoch %d]; '
        best_log = ''

                
        for loss_type, (curr_best_epoch, curr_best_loss) in best_epoch_losses.items():

            if (val_loss_meters[loss_type].avg < curr_best_loss and loss_type != 'f1') or \
                    (val_loss_meters[loss_type].avg > curr_best_loss and loss_type == 'f1'):
                best_epoch_losses[loss_type] = (curr_epoch_num, val_loss_meters[loss_type].avg)
                # save_model('%s/best_%s_network.pth' % (log_dir, loss_type), net)
                save_model('%s/best_%s_network_refinement_16384.pth' % (log_dir, loss_type), net2)
                if loss_type == 'cd_t' and best_epoch_losses[loss_type][1]<0.00062:
                    save_model('%s/best_%s_network_refinement_16384_%s.pth' % (log_dir, loss_type,str(best_epoch_losses[loss_type][1])), net2)
                logging.info('Best %s net saved!' % loss_type)
                best_log += fmt % (loss_type, best_epoch_losses[loss_type][1], best_epoch_losses[loss_type][0])
            else:
                best_log += fmt % (loss_type, curr_best_loss, curr_best_epoch)

        curr_log = ''
        for loss_type, meter in val_loss_meters.items():
            curr_log += 'curr_%s: %f; ' % (loss_type, meter.avg)

        logging.info(curr_log)
        logging.info(best_log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train config file')
    parser.add_argument('-c', '--config', help='path to config file', required=True)
    arg = parser.parse_args()
    config_path = arg.config
    args = munch.munchify(yaml.safe_load(open(config_path)))

    time = datetime.datetime.now().isoformat()[:19]
    if args.load_model:
        exp_name = os.path.basename(os.path.dirname(args.load_model))
        log_dir = os.path.dirname(args.load_model)
    else:
        exp_name = args.model_name + '_' + args.loss + '_' + args.flag + '_' + time
        log_dir = os.path.join(args.work_dir, exp_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(os.path.join(log_dir, 'train.log')),
                                                      logging.StreamHandler(sys.stdout)])
    train()



