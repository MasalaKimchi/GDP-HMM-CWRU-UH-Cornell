import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch
import torch.nn as nn
import torch.optim as optim
from nnunet_mednext import create_mednext_v1
import data_loader_efficient as data_loader
import yaml
import argparse
import os

class GDPLightningModel(pl.LightningModule):
    def __init__(self, cfig):
        super(GDPLightningModel, self).__init__()
        self.model = create_mednext_v1(
            num_input_channels=cfig['model_params']['num_input_channels'],
            num_classes=cfig['model_params']['out_channels'],
            model_id=cfig['model_params']['model_id'],
            kernel_size=cfig['model_params']['kernel_size'],
            deep_supervision=cfig['model_params']['deep_supervision']
        )
        self.criterion = nn.SmoothL1Loss(beta=0.25)
        self.l1 = nn.L1Loss()
        self.lr = cfig['lr']
        self.num_epochs = cfig['num_epochs']
        self.cfig = cfig
        self.sig_act = nn.Sigmoid()
    
    def training_step(self, batch, batch_idx):
        inputs, labels = batch['data'], batch['label']
        outputs = self.model(inputs) 
        if self.cfig['act_sig']:
            outputs = self.sig_act(outputs.clone())

        loss = self.criterion(outputs * self.cfig['scale_out'], labels) * self.cfig['scale_loss']
        l1_loss = self.l1(outputs * self.cfig['scale_out'], labels) * self.cfig['scale_loss']
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.cfig['loader_params']['train_bs'])
        self.log('train_l1_loss', l1_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.cfig['loader_params']['train_bs'])

        if self.current_epoch % 10 == 0 and batch_idx == 3:
            os.system('nvidia-smi')

        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch['data'], batch['label']
        outputs = self.model(inputs) 
        if self.cfig['act_sig']:
            outputs = self.sig_act(outputs.clone())
            
        loss = self.criterion(outputs * self.cfig['scale_out'], labels) * self.cfig['scale_loss']
        l1_loss = self.l1(outputs * self.cfig['scale_out'], labels) * self.cfig['scale_loss']
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.cfig['loader_params']['train_bs'])
        self.log('val_l1_loss', l1_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.cfig['loader_params']['train_bs'])
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam([{'params': self.model.parameters(), 'initial_lr': self.lr}], lr=self.lr)
        scheduler = {'scheduler': optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= self.num_epochs), 
                     'interval': 'epoch', 'frequency': 1}
        return [optimizer], [scheduler]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('cfig_path', type=str)
    parser.add_argument('--phase', default='train', type=str)
    args = parser.parse_args()

    cfig = yaml.load(open(args.cfig_path), Loader=yaml.FullLoader)

    # Data Loaders
    loaders = data_loader.GetLoader(cfig=cfig['loader_params'])
    train_loader = loaders.train_dataloader()
    val_loader = loaders.val_dataloader()

    # Model
    model = GDPLightningModel(cfig)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"   
    if torch.cuda.device_count() > 1:
        # stratgy = 'ddp_find_unused_parameters_true'
        stratgy='ddp'
        sync_batchnorm = True
        use_distributed_sampler = True
        
    else:
        stratgy = 'auto' 
        sync_batchnorm = False
        use_distributed_sampler = False

    # Callbacks
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfig['save_model_root'],
        filename='mednextLK5-SmoothL1-{epoch:02d}-{val_loss:.4f}-{val_l1_loss:.4f}',
        save_top_k=2,
        monitor='val_loss',
        mode='min', 
        save_last=False
    )

    mylogger = CSVLogger(cfig['save_model_root'])
    save_model_weight_dir = mylogger.log_dir

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfig['num_epochs'],
        devices = 'auto', 
        accelerator=accelerator, strategy=stratgy, sync_batchnorm=sync_batchnorm,
        use_distributed_sampler=use_distributed_sampler, 
        logger=mylogger, 
        default_root_dir=save_model_weight_dir,
        callbacks=[lr_monitor, checkpoint_callback]
    )

    # Training
    trainer.fit(model, train_loader, val_loader)