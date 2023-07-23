from tqdm import trange
import torch

from torch.utils.data import DataLoader

from logger import Logger
from modules.model import GeneratorFullModel, DiscriminatorFullModel
import modules.model as MODEL
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.parallel import DistributedDataParallel as DDP
import pdb
from sync_batchnorm import DataParallelWithCallback
from evaluation.evaluation_dataset import EvaluationDataset
import numpy as np
from frames_dataset import DatasetRepeater
import torchvision.utils as vutils
import os

class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
def printGrad(net):
    for name,params in net.named_parameters():
        print("==>name: ", name, " ==>grad_requires: ",params.requires_grad," ==>max grad: ", params.grad.max(), " ==>min grad: ", params.grad.min()," ==>mean grad: ", params.grad.mean())
def CheckGrad(dic):
    for name,params in dic:
        print("==>name: ", name, " ==>grad_requires: ",params.requires_grad," ==>grad: ", params.grad.mean())
def train(config, generator, discriminator, kp_detector, checkpoint, log_dir, dataset, rank,device,opt,writer):
    train_params = config['train_params']

    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=train_params['lr_generator'], betas=(0.5, 0.999))
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=train_params['lr_discriminator'], betas=(0.5, 0.999))
    optimizer_kp_detector = torch.optim.Adam(kp_detector.parameters(), lr=train_params['lr_kp_detector'], betas=(0.5, 0.999))

    if checkpoint is not None:
        start_epoch = Logger.load_cpk(checkpoint, generator, discriminator, kp_detector,
                                        optimizer_generator, optimizer_discriminator,
                                        None if train_params['lr_kp_detector'] == 0 else optimizer_kp_detector)
    else:
        start_epoch = 0
    scheduler_generator = MultiStepLR(optimizer_generator, train_params['epoch_milestones'], gamma=train_params['gamma'],
                                      last_epoch=start_epoch - 1)
    scheduler_discriminator = MultiStepLR(optimizer_discriminator, train_params['epoch_milestones'], gamma=train_params['gamma'],
                                          last_epoch=start_epoch - 1)
    scheduler_kp_detector = MultiStepLR(optimizer_kp_detector, train_params['epoch_milestones'], gamma=train_params['gamma'],
                                        last_epoch=-1 + start_epoch * (train_params['lr_kp_detector'] != 0))

    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])
    sampler = torch.utils.data.distributed.DistributedSampler(dataset,num_replicas=torch.cuda.device_count(),rank=rank)
    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=False, num_workers=8, sampler=sampler, drop_last=True)
    
    generator_full = getattr(MODEL,opt.GFM)(kp_detector, generator, discriminator, train_params,opt)
    discriminator_full = DiscriminatorFullModel(kp_detector, generator, discriminator, train_params)

    # if torch.cuda.is_available():
    #     generator_full = DataParallelWithCallback(generator_full, device_ids=device_ids)
    #     discriminator_full = DataParallelWithCallback(discriminator_full, device_ids=device_ids)

    test_dataset = EvaluationDataset(dataroot='/data/fhongac/origDataset/vox1_frames',size = [512,512],pairs_list='data/vox_evaluation_v2.csv')
    test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size = 2,
            shuffle=False,
            num_workers=4)
    
    #copy net_g weight
    # ema = EMA(generator, decay=0.5**(32 / (10 * 1000)))
    # ema.register()

    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq']) as logger:
        
        for epoch in trange(start_epoch, train_params['num_epochs']):
            #parallel
            sampler.set_epoch(epoch)
            total = len(dataloader)
            epoch_train_loss = 0
            generator.train(), discriminator.train(), kp_detector.train()
           
            with tqdm(total=total) as par:
                for i,x in enumerate(dataloader):
                    # print(generator.module.gau.to_q_gate[0].weight())
                    x['source'] = x['source'].to(device)
                    x['driving'] = x['driving'].to(device)
                    if opt.linear_grow_mb_weight:
                        weight = (total*epoch+i)/5000 if (total*epoch+i)/5000<1 else 1
                    else:
                        weight = 1
                    losses_generator, generated = generator_full(x,weight,epoch=epoch) 
                    # print(generated['Fwarp'].mean().item(),generated['Fwarp'].min().item(),generated['Fwarp'].max().item())
                    loss_values = [val.mean() for val in losses_generator.values()]
                    loss = sum(loss_values)
                    loss.backward()
                    if not torch.isfinite(loss).item():
                        optimizer_generator.zero_grad()
                        optimizer_kp_detector.zero_grad()
                        print('NaN=============')
                    else:
                        # printGrad(generator)
                        # printGrad(kp_detector)
                        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=10, norm_type=2)
                        torch.nn.utils.clip_grad_norm_(kp_detector.parameters(), max_norm=10, norm_type=2)
                        optimizer_generator.step()
                        optimizer_generator.zero_grad()
                        optimizer_kp_detector.step()
                        optimizer_kp_detector.zero_grad()
                    epoch_train_loss+=loss.item()
                    if train_params['loss_weights']['generator_gan'] != 0:
                        optimizer_discriminator.zero_grad()
                        losses_discriminator = discriminator_full(x, generated)
                        loss_values = [val.mean() for val in losses_discriminator.values()]
                        loss = sum(loss_values)
                        loss.backward()
                        optimizer_discriminator.step()
                        optimizer_discriminator.zero_grad()
                    else:
                        losses_discriminator = {}
                    losses_generator.update(losses_discriminator)
                    losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
                    for k,v in losses.items():
                        writer.add_scalar(k, v, total*epoch+i)
                    logger.log_iter(losses=losses)
                    # generator_full.updateMB()
                    par.update(1)
                    
            epoch_train_loss = epoch_train_loss/total
            if (epoch + 1) % train_params['checkpoint_freq'] == 0:
                # ema.apply_shadow()
                # # evaluate
                # ema.restore()
                writer.add_scalar('epoch_train_loss', epoch_train_loss, epoch)
                try:
                    torch.save(generator_full.mb.mb_item, os.path.join(log_dir, '%s-mb.pt' % str(epoch).zfill(8)))
                except Exception as e:
                    print(e)
            scheduler_generator.step()
            scheduler_discriminator.step()
            scheduler_kp_detector.step()
            logger.log_epoch(epoch, {'generator': generator,
                                     'discriminator': discriminator,
                                     'kp_detector': kp_detector,
                                     'optimizer_generator': optimizer_generator,
                                     'optimizer_discriminator': optimizer_discriminator,
                                     'optimizer_kp_detector': optimizer_kp_detector}, inp=x, out=generated)
           
            generator.eval(), discriminator.eval(), kp_detector.eval()
            
            if False and (epoch + 1) % train_params['checkpoint_freq'] == 0:
                epoch_eval_loss = 0
                for i, data in tqdm(enumerate(test_dataloader)):
                    data['source'] = data['source'].cuda()
                    data['driving'] = data['driving'].cuda()
                    losses_generator, generated = generator_full(data) 
                    loss_values = [val.mean() for val in losses_generator.values()]
                    loss = sum(loss_values)
                    epoch_eval_loss+=loss.item()
                epoch_eval_loss = epoch_eval_loss/len(test_dataloader)
                writer.add_scalar('epoch_eval_loss', epoch_eval_loss, epoch)

                try:
                    source = data['source'][0:1]
                    driving = data['driving'][0:1]
                    prediction = generated['prediction'][0:1]
                    rst = torch.cat((source,driving,prediction),0)
                    img_grid = vutils.make_grid(rst, normalize=True, scale_each=True, nrow=3)  # normalize进行归一化处理
                    writer.add_image("source-driving-prediction",img_grid, global_step=epoch,dataformats='CHW')
                except Exception as e:
                    print(e)  
                for key in generated:
                    if 'visual' in key:
                        try:
                            value = generated[key]
                            l = len(value.shape)
                            if l==2:
                                n,d=value.shape
                                visualization = value.view(n,1,int(np.sqrt(d)),int(np.sqrt(d)))
                            if l==4:
                                bs,c,w,h=value.shape
                                visualization = value[0:1].permute(1,0,2,3)
                            img_grid = vutils.make_grid(visualization, normalize=True, scale_each=True, nrow=16)  # normalize进行归一化处理
                            writer.add_image(key, img_grid, global_step=epoch)
                            print("Tensorboard saves {}".format(key))
                        except Exception as e:
                            print(e)
               