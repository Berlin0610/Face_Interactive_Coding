from tqdm import trange
import torch

from torch.utils.data import DataLoader

from logger import Logger
from modules.model import GeneratorFullModel, DiscriminatorFullModel

from torch.optim.lr_scheduler import MultiStepLR

from sync_batchnorm import DataParallelWithCallback

from frames_dataset import DatasetRepeater
torch.backends.cudnn.benchmark = False



def train(config, generator, discriminator, checkpoint, log_dir, dataset, device_ids):    
    train_params = config['train_params']

    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=train_params['lr_generator'], betas=(0.5, 0.999))
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=train_params['lr_discriminator'], betas=(0.5, 0.999))

    if checkpoint is not None:
        start_epoch = Logger.load_cpk(checkpoint, generator, discriminator,
                                      optimizer_generator, optimizer_discriminator)
   
    else:
        start_epoch = 0

    scheduler_generator = MultiStepLR(optimizer_generator, train_params['epoch_milestones'], gamma=0.1,
                                      last_epoch=start_epoch - 1)
    scheduler_discriminator = MultiStepLR(optimizer_discriminator, train_params['epoch_milestones'], gamma=0.1,
                                          last_epoch=start_epoch - 1)



    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])

    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=16, drop_last=True)
    
    
    
    generator_full = GeneratorFullModel(generator, discriminator, train_params)
    discriminator_full = DiscriminatorFullModel(generator, discriminator, train_params)

    # if torch.cuda.is_available():
    #     generator_full = DataParallelWithCallback(generator_full, device_ids=device_ids)
    #     discriminator_full = DataParallelWithCallback(discriminator_full, device_ids=device_ids)
        
    if torch.cuda.is_available():
        generator_full = DataParallelWithCallback(generator_full).cuda()
        discriminator_full = DataParallelWithCallback(discriminator_full).cuda()      
        

    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):
            for x in dataloader:
                losses_generator, generated = generator_full(x)

                loss_values = [val.mean() for val in losses_generator.values()]
                loss = sum(loss_values)

                loss.backward()
                optimizer_generator.step()
                optimizer_generator.zero_grad()


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
                logger.log_iter(losses=losses)

            scheduler_generator.step()
            scheduler_discriminator.step()
            
            logger.log_epoch(epoch, {'generator': generator,
                                     'discriminator': discriminator,
                                     'optimizer_generator': optimizer_generator,
                                     'optimizer_discriminator': optimizer_discriminator}, inp=x, out=generated)