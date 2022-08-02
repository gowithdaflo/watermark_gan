from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision.ops.focal_loss import sigmoid_focal_loss
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import torch
from .metrics import Metrics
from ..utils.utils import float_to_uint8
from .vgg_loss import VGGLoss
from functools import partial
import logging

class Trainer:
    def __init__(self, 
                generator, 
                discriminator, 
                dataset,
                batch_size,
                learning_rate,
                log_dir,
                pretrain = True,
                device="cpu"
                ):
        
        self.device = device
        self.batch_size = batch_size
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        
        self.dataset = dataset
        self.data_loader = DataLoader(dataset, 
                                      batch_size=batch_size, 
                                      drop_last=False,
                                      num_workers=2,
                                      pin_memory=True)
        logging.info(f"nBatches in dataset: {len(self.data_loader)}")
        
        self._load_test_samples()
        
        self.optimG = optim.Adam(self.generator.parameters(), lr=learning_rate, betas=(0.9, 0.999))#, weight_decay=weight_decayG)
        self.optimD = optim.Adam(self.discriminator.parameters(), lr=learning_rate, betas=(0.9, 0.999))#, weight_decay=weight_decayG)
        
        self.bce_logits_loss = torch.nn.BCEWithLogitsLoss() # − [y * log(x) + (1−y) * log(1−x)]
        # self.bce_logits_loss = partial(sigmoid_focal_loss, alpha=0.25, gamma=2, reduction="mean")
        self.gan_weight = 1e-3
        self.vgg_weight = 2e-5 # rescale vgg fmaps to be of the same scale as mse loss
        self.l1_weight = 1
        self.vgg_loss = VGGLoss(device=device)
        self.l1_loss = torch.nn.L1Loss()
        self.mask_weight = 1.0
            
        self.metrics = Metrics(["lossG", "lossD", "mask_miou"])
        self.summary_writer = SummaryWriter(log_dir)
        
        self._step = 0
        self._pretrain_mode = pretrain
        logging.info(f"Pretrain mode: {self._pretrain_mode}")
        
    def _load_test_samples(self):
        self.test_samples = {"inputs": [], "targets": [], "mask": []}
        for i in range(8):
            sample = self.dataset[i]
            for key in self.test_samples.keys():
                self.test_samples[key].append( sample[key] )
        for key, value in self.test_samples.items():
            self.test_samples[key] = torch.stack(value, dim=0).to(self.device)
        
    def set_pretrain_mode(self, pretrain):
        logging.info(f"Set pretrain mode to: {pretrain}")
        self._pretrain_mode = pretrain
        
    def generator_loss(self, gen_fake, dis_fake, target):
        # classify generated output as real
        if self._pretrain_mode:
            gan_loss = 0.0
        else:    
            gan_loss = self.gan_weight * self.bce_logits_loss(dis_fake, torch.ones_like(dis_fake))
        l1_loss = self.l1_weight * self.l1_loss(gen_fake, target)
        vgg_loss = self.vgg_weight * self.vgg_loss(gen_fake, target)
        return gan_loss + l1_loss + vgg_loss
    
    def discriminator_loss(self, dis_real, dis_fake):
        # classify real images and fakes properly
        real_loss = self.bce_logits_loss(dis_real, torch.ones_like(dis_real))
        generated_loss = self.bce_logits_loss(dis_fake, torch.zeros_like(dis_fake))
        return real_loss + generated_loss

    def train_on_batch(self, sample):
        self._step+=1
        sample = {k: v.to(self.device) for k, v in sample.items()}
        inputs, targets, mask = sample["inputs"], sample["targets"], sample["mask"]
        # inputs, targets, mask, recep_mask = sample["inputs"], sample["targets"], sample["mask"], sample["recep_mask"]

        self.generator.train()
        self.discriminator.train()
        
        gen_fake, gen_mask = self.generator(inputs)
        
        if self._pretrain_mode:
            dis_fake = torch.tensor(0.0)
            lossD = torch.tensor(0.0)
        else:
            dis_real = self.discriminator(inputs, targets)
            dis_fake = self.discriminator(gen_fake.detach(), targets)

            ### ------------------------ Train Discriminator ------------------------ ###
            self.discriminator.zero_grad()
            lossD = self.discriminator_loss(dis_real, dis_fake)
            lossD.backward()
            self.optimD.step()

        ### -------------------------- Train Generator -------------------------- ###
        self.generator.zero_grad()
        lossG = self.generator_loss(gen_fake, dis_fake.detach(), targets)
        lossG += self.mask_weight * self.bce_logits_loss(gen_mask, mask.double()) # mask loss
        
        lossG.backward()
        self.optimG.step()

        with torch.no_grad():
            gen_mask = gen_mask > 0
            mask_miou = (gen_mask * mask).sum() / (gen_mask + mask).sum()
        
        self.metrics.update_state(inputs.shape[0],
                                lossG = lossG.detach(),
                                lossD = lossD.detach(),
                                mask_miou = mask_miou.detach()
                                )
            
    def train_epoch(self):
        for sample in self.data_loader:
            self.train_on_batch(sample)
        self.on_epoch_end()
        
    def test_on_batch(self, samples=None):
        """ 
        Args:
            noise: torch.Tensor
                Generate fake images from this noise.
            Nmax: int, optional
                Maximum number of images to save.
        """
        self.generator.eval()
        
        with torch.no_grad():
            if samples:
                samples = {k: v.to(self.device) for k, v in samples.items()}
                
            inputs = samples["inputs"]
            targets = samples["targets"]
            mask = (samples["mask"] * 2 - 1).repeat(1,3,1,1)

            fake, pred_mask = self.generator(inputs)
            pred_mask = (torch.sigmoid(pred_mask) * 2 - 1).repeat(1,3,1,1)
            
        merged = float_to_uint8(torch.cat([inputs, targets, fake, pred_mask, mask], dim=3))
        img_grid_merged = make_grid(merged, normalize=False, nrow=1)
        self.summary_writer.add_image(
            "Image - Target - Pred - Pred Mask - Mask", img_grid_merged, global_step=self._step
            )  
        return merged
            
    def on_epoch_end(self):
        """
        """
        epoch = int(self._step// (len(self.dataset)//self.batch_size) )
        
        result = self.metrics.result()
        self.metrics.write(self.summary_writer, self._step)
        self.metrics.reset_states()

        logging.info(f"Epoch {epoch}: " + ", ".join([f"{key}: {result[key]:.4f}" for key in self.metrics.keys]))
