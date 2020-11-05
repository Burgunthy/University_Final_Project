import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from skimage.transform import resize

from model import *
from dataset import *
from util import *

import matplotlib.pyplot as plt

from torchvision import transforms


def train(args):
    ## 트레이닝 파라메터 설정하기
    mode = args.mode
    train_continue = args.train_continue

    code_size = args.code_size
    init_size = args.init_size
    lr = args.lr
    batch_size = args.batch_size
    num_iteration = args.num_iteration
    n_critic = args.n_critic

    data_dir = args.data_dir
    ckpt_dir = args.ckpt_dir
    log_dir = args.log_dir
    result_dir = args.result_dir

    wgt = args.wgt

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("mode: %s" % mode)

    print("code_size: %d" % code_size)
    print("init_size: %d" % init_size)
    print("learning rate: %.4e" % lr)
    print("batch size: %d" % batch_size)
    print("iteration: %d" % num_iteration)
    print("n_critic: %d" % n_critic)

    print("data dir: %s" % data_dir)
    print("ckpt dir: %s" % ckpt_dir)
    print("log dir: %s" % log_dir)
    print("result dir: %s" % result_dir)

    print("device: %s" % device)

    ## 디렉토리 생성하기
    result_dir_train = os.path.join(result_dir, 'train')
    result_dir_val = os.path.join(result_dir, 'val')

    if not os.path.exists(result_dir_train):
        os.makedirs(os.path.join(result_dir_train, 'jpg'))

    if not os.path.exists(result_dir_val):
        os.makedirs(os.path.join(result_dir_val, 'jpg'))

    ## 네트워크 학습하기

    ## 네트워크 생성하기
    netG = Generator(code_size).to(device)
    netD = Discriminator().to(device)

    ## 손실함수 정의하기
    # fn_loss = nn.BCEWithLogitsLoss().to(device)
    # fn_loss = nn.MSELoss().to(device)

    fn_l1 = nn.L1Loss().to(device)
    fn_gan = nn.BCELoss().to(device)
    #  fn_gan = nn.L1Loss().to(device)

    ## Optimizer 설정하기
    optimG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    optimD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

    ## 그밖에 부수적인 functions 설정하기
    fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    fn_denorm = lambda x, mean, std: (x * std) + mean
    fn_class = lambda x: 1.0 * (x > 0.5)

    cmap = None

    ## Tensorboard 를 사용하기 위한 SummaryWriter 설정
    writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
    writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

    ## 네트워크 학습시키기
    iteration = 0
    i_that = 0
    epoch = 0

    # TRAIN MODE
    if mode == 'train':
        if train_continue == "on":
            netG, netD, optimG, optimD, i_that = load(ckpt_dir=ckpt_dir,
                                                        netG=netG, netD=netD,
                                                        optimG=optimG, optimD=optimD)
        step = 5

        dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), code_size=code_size, step=step)
        loader_train = DataLoader(dataset_train, batch_size=batch_size,
                                  shuffle=True, num_workers=8)

        dataset = iter(loader_train)
        data = next(dataset)

        dataset_mask = mask_Dataset(data_dir=os.path.join(data_dir, 'mask'), code_size=code_size, step=step)
        loader_mask = DataLoader(dataset_mask, batch_size=batch_size,
                                              shuffle=True, num_workers=8)
        mask_dataset = iter(loader_mask)
        mask = next(mask_dataset)

        dataset_val = Dataset(data_dir=os.path.join(data_dir, 'val'), code_size=code_size, step=step)
        loader_val = DataLoader(dataset_val, batch_size=batch_size,
                                  shuffle=True, num_workers=8)

        dataset_mask_val = mask_Dataset(data_dir=os.path.join(data_dir, 'mask_val'), code_size=code_size, step=step)
        loader_mask_val = DataLoader(dataset_mask_val, batch_size=batch_size,
                                              shuffle=True, num_workers=8)



        pbar = tqdm(range(30000))
        vbar = tqdm(range(40))

        netG.train()
        netD.train()

        loss_G_l1_train = []
        loss_G_gan_train = []
        loss_D_real_train = []
        loss_D_fake_train = []

        for i in pbar:

            if iteration > 5000:     # 높은 pixel 계산
                iteration = 0
                step += 1

                if step > 5:          # 최대 step 5 설정
                    step = 5

                dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), code_size=code_size, step=step)
                loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)
                dataset = iter(loader_train)
                data = next(dataset)

                dataset_mask = mask_Dataset(data_dir=os.path.join(data_dir, 'mask'), code_size=code_size, step=step)
                loader_mask = DataLoader(dataset_mask, batch_size=batch_size, shuffle=True, num_workers=8)
                mask_dataset = iter(loader_mask)
                mask = next(mask_dataset)

                dataset_val = Dataset(data_dir=os.path.join(data_dir, 'val'), code_size=code_size, step=step)
                loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=8)


                dataset_mask_val = mask_Dataset(data_dir=os.path.join(data_dir, 'mask_val'), code_size=code_size, step=step)
                loader_mask_val = DataLoader(dataset_mask_val, batch_size=batch_size, shuffle=True, num_workers=8)


            # forward pass
            try:
                data = next(dataset)
                mask = next(mask_dataset)

            except (OSError, StopIteration):
                epoch += 1
                writer_train.add_scalar('loss_G_l1', np.mean(loss_G_l1_train), epoch)
                writer_train.add_scalar('loss_G_gan', np.mean(loss_G_gan_train), epoch)
                writer_train.add_scalar('loss_D_real', np.mean(loss_D_real_train), epoch)
                writer_train.add_scalar('loss_D_fake', np.mean(loss_D_fake_train), epoch)
                #val training 진행

                with torch.no_grad():
                    netG.eval()
                    netD.eval()

                    loss_G_l1_val = []
                    loss_G_gan_val = []
                    loss_D_real_val = []
                    loss_D_fake_val = []

                    dataset_val = iter(loader_val)
                    mask_dataset_val = iter(loader_mask_val)
                    
                    for j in vbar:
                        # forward pass
                        data = next(dataset_val)
                        mask = next(mask_dataset_val)

                        label = data['label'].to(device)
                        input = data['input'].to(device)

                        mask_input = mask['input'].to(device)
                        msk1 = mask['mask1'].to(device)
                        msk2 = mask['mask2'].to(device)

                        input = mask_input * input

                        output = netG(input, step=step)
                        output = (msk2 * output) + (msk1 * label)

                        pred_real = netD(label, step=step)
                        pred_fake = netD(output.detach(), step=step)

                        loss_D_real = fn_gan(pred_real, torch.ones_like(pred_real))
                        loss_D_fake = fn_gan(pred_fake, torch.zeros_like(pred_fake))
                        loss_D = 0.5 * (loss_D_real + loss_D_fake)

                        pred_fake = netD(output, step=step)  # 같이 넣어준다

                        loss_G_gan = fn_gan(pred_fake, torch.ones_like(pred_fake))
                        loss_G_l1 = fn_l1(output, label)
                        loss_G = loss_G_gan + wgt * loss_G_l1


                        loss_G_l1_val += [loss_G_l1.item()]
                        loss_G_gan_val += [loss_G_gan.item()]
                        loss_D_real_val += [loss_D_real.item()]
                        loss_D_fake_val += [loss_D_fake.item()]


                    #writer_val.add_image('input', input, epoch, dataformats='NHWC')
                    #writer_val.add_image('label', label, epoch, dataformats='NHWC')
                    #writer_val.add_image('output', output, epoch, dataformats='NHWC')

                    writer_val.add_scalar('loss_G_l1', np.mean(loss_G_l1_val), epoch)
                    writer_val.add_scalar('loss_G_gan', np.mean(loss_G_gan_val), epoch)
                    writer_val.add_scalar('loss_D_real', np.mean(loss_D_real_val), epoch)
                    writer_val.add_scalar('loss_D_fake', np.mean(loss_D_fake_val), epoch)

                netG.train()
                netD.train()

                loss_G_l1_train = []
                loss_G_gan_train = []
                loss_D_real_train = []
                loss_D_fake_train = []

                dataset = iter(loader_train)
                mask_dataset = iter(loader_mask)

                data = next(dataset)
                mask = next(mask_dataset)

            label = data['label'].to(device)
            input = data['input'].to(device)

            mask_input = mask['input'].to(device)
            msk1 = mask['mask1'].to(device)
            msk2 = mask['mask2'].to(device)
            
            iteration += 1
            input = mask_input * input

            output = netG(input, step=step)
            output = (msk2 * output) + (msk1 * label)
            #output 기존 이미지랑 합치기

            # backward netD
            set_requires_grad(netD, True)
            optimD.zero_grad()

            # real = torch.cat([input, label], dim=1)
            # fake = torch.cat([input, output], dim=1)

            pred_real = netD(label, step=step)
            pred_fake = netD(output.detach(), step=step)

            loss_D_real = fn_gan(pred_real, torch.ones_like(pred_real))
            loss_D_fake = fn_gan(pred_fake, torch.zeros_like(pred_fake))
            loss_D = 0.5 * (loss_D_real + loss_D_fake)

            loss_D.backward()
            optimD.step()

                # backward netG
            set_requires_grad(netD, False)
            optimG.zero_grad()

                # fake = torch.cat([input, output], dim=1)
            pred_fake = netD(output, step=step)  # 같이 넣어준다

            loss_G_gan = fn_gan(pred_fake, torch.ones_like(pred_fake))
            loss_G_l1 = fn_l1(output, label)
            loss_G = loss_G_gan + wgt * loss_G_l1

            loss_G.backward()
            optimG.step()

                # 손실함수 계산
            loss_G_l1_train += [loss_G_l1.item()]
            loss_G_gan_train += [loss_G_gan.item()]
            loss_D_real_train += [loss_D_real.item()]
            loss_D_fake_train += [loss_D_fake.item()]

            print("TRAIN: i %04d | step %04d | epoch %04d |"
                      "GEN L1 %.4f | GEN GAN %.4f | "
                      "DISC REAL: %.4f | DISC FAKE: %.4f" %
                      (i, step, epoch,
                       np.mean(loss_G_l1_train), np.mean(loss_G_gan_train),
                       np.mean(loss_D_real_train), np.mean(loss_D_fake_train)))

            

            if (i + 1) % 1250 == 0:
                    # Tensorboard 저장하기
                input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5)).squeeze()
                label = fn_tonumpy(fn_denorm(label, mean=0.5, std=0.5)).squeeze()
                output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5)).squeeze()

                input = np.clip(input, a_min=0, a_max=1)
                label = np.clip(label, a_min=0, a_max=1)
                output = np.clip(output, a_min=0, a_max=1)

                id = i + i_that + 1

                plt.imsave(os.path.join(result_dir_train, 'jpg', '%04d_input.png' % id), input[0], cmap=cmap)
                plt.imsave(os.path.join(result_dir_train, 'jpg', '%04d_label.png' % id), label[0], cmap=cmap)
                plt.imsave(os.path.join(result_dir_train, 'jpg', '%04d_output.png' % id), output[0], cmap=cmap)

                writer_train.add_image('input', input, id, dataformats='NHWC')
                writer_train.add_image('label', label, id, dataformats='NHWC')
                writer_train.add_image('output', output, id, dataformats='NHWC')

            if (i + 1) % 2500 == 0:
                save(ckpt_dir=ckpt_dir, netG=netG, netD=netD, optimG=optimG, optimD=optimD, i=i + i_that + 1)
            


        writer_train.close()
        writer_val.close()


def test(args):
    ## 트레이닝 파라메터 설정하기
    mode = args.mode
    train_continue = args.train_continue

    code_size = args.code_size
    init_size = args.init_size
    lr = args.lr
    batch_size = args.batch_size
    num_iteration = args.num_iteration
    n_critic = args.n_critic

    data_dir = args.data_dir
    ckpt_dir = args.ckpt_dir
    log_dir = args.log_dir
    result_dir = args.result_dir

    wgt = args.wgt

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("mode: %s" % mode)

    print("code_size: %d" % code_size)
    print("init_size: %d" % init_size)
    print("learning rate: %.4e" % lr)
    print("batch size: %d" % batch_size)
    print("iteration: %d" % num_iteration)
    print("n_critic: %d" % n_critic)

    print("data dir: %s" % data_dir)
    print("ckpt dir: %s" % ckpt_dir)
    print("log dir: %s" % log_dir)
    print("result dir: %s" % result_dir)

    print("device: %s" % device)

    ## 디렉토리 생성하기
    result_dir_test = os.path.join(result_dir, 'test')

    if not os.path.exists(result_dir_test):
        os.makedirs(os.path.join(result_dir_test, 'jpg'))
        os.makedirs(os.path.join(result_dir_test, 'numpy'))

    ## 네트워크 학습하기


    ## 네트워크 생성하기
    netG = Generator(code_size).to(device)
    netD = Discriminator().to(device)

    ## 손실함수 정의하기
    fn_l1 = nn.L1Loss().to(device)

    ## Optimizer 설정하기
    optimG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    optimD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

    ## 그밖에 부수적인 functions 설정하기
    fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    fn_denorm = lambda x, mean, std: (x * std) + mean
    fn_class = lambda x: 1.0 * (x > 0.5)

    cmap = None

    ## 네트워크 학습시키기
    i_that = 0
    number = 15

    # TRAIN MODE
    if mode == "test":
        netG, netD, optimG, optimD, i_that = load(ckpt_dir=ckpt_dir, netG=netG, netD=netD, optimG=optimG,
                                                    optimD=optimD)

        step = 5

        dataset_test = Dataset(data_dir=os.path.join(data_dir, 'test'), code_size=code_size, step=step)
        loader_test = DataLoader(dataset_test, batch_size=batch_size,
                                 shuffle=False, num_workers=0)
        dataset_mask = mask_Dataset(data_dir=os.path.join(data_dir, 'mask_test'), code_size=code_size, step=step)
        loader_mask = DataLoader(dataset_mask, batch_size=batch_size, shuffle=False, num_workers=0)
        mask_dataset = iter(loader_mask)
        mask = next(mask_dataset)

        # 그밖에 부수적인 variables 설정하기
        num_data_test = len(dataset_test)
        num_batch_test = np.ceil(num_data_test / batch_size)

        with torch.no_grad():
            netG.eval()

            for batch, data in enumerate(loader_test, 1):
                # forward pass
                label = data['label'].to(device)
                input = data['input'].to(device)

                # mask = next(mask_dataset)

                mask_input = mask['input'].to(device)
                msk1 = mask['mask1'].to(device)
                msk2 = mask['mask2'].to(device)
            
                input = mask_input * input

                output = netG(input, step=step)
                output = (msk2 * output) + (msk1 * label)

                # Tensorboard 저장하기
                input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5)).squeeze()
                label = fn_tonumpy(fn_denorm(label, mean=0.5, std=0.5)).squeeze()
                output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5)).squeeze()

                input = np.clip(input, a_min=0, a_max=1)
                label = np.clip(label, a_min=0, a_max=1)
                output = np.clip(output, a_min=0, a_max=1)

                id = number
                number += 1

                plt.imsave(os.path.join(result_dir_test, 'jpg', '%04d_input.png' % id), input, cmap=cmap)
                plt.imsave(os.path.join(result_dir_test, 'jpg', '%04d_label.png' % id), label, cmap=cmap)
                plt.imsave(os.path.join(result_dir_test, 'jpg', '%04d_output.png' % id), output, cmap=cmap)