import dataclasses
import types

import accelerate
import torch
import torch.nn.functional as F


@dataclasses.dataclass
class VanillaGAN:
    cfg: types.SimpleNamespace
    generator: torch.nn.Module
    discriminator: torch.nn.Module
    generator_opt: torch.optim.Optimizer    
    discriminator_opt: torch.optim.Optimizer
    accelerator: accelerate.Accelerator

    @property
    def _batch_size(self) -> int:
        return self.cfg.bs

    def _step_discriminator(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> tuple[torch.Tensor, float, float]:
        self.generator.eval()

        d_real, d_fake = self.discriminator(real_data), self.discriminator(fake_data)

        device = real_data.device
        batch_size = real_data.size(0)
        real_labels = torch.ones((batch_size, 1), device=device) * (1 - self.cfg.smooth)
        fake_labels = torch.ones((batch_size, 1), device=device) * self.cfg.smooth

        disc_loss_real = F.binary_cross_entropy_with_logits(d_real, real_labels)
        disc_loss_fake = F.binary_cross_entropy_with_logits(d_fake, fake_labels)
        disc_loss = (disc_loss_real + disc_loss_fake) / 2

        disc_acc_real = (torch.sigmoid(d_real) < 0.5).float().mean().item()
        disc_acc_fake = (torch.sigmoid(d_fake) > 0.5).float().mean().item()
        self.generator.train()
        self.discriminator_opt.zero_grad()
        self.accelerator.backward(
            disc_loss * self.cfg.loss_coefficient_disc)
        self.accelerator.clip_grad_norm_(
            self.discriminator.parameters(), 
            self.cfg.max_grad_norm
        )
        self.discriminator_opt.step()
        return disc_loss, disc_acc_real, disc_acc_fake
    
    def _step_generator(self, fake_data: torch.Tensor) -> tuple[torch.Tensor, float]:
        self.discriminator.eval()
        d_fake = self.discriminator(fake_data)
        device = fake_data.device
        batch_size = fake_data.size(0)
        real_labels = torch.zeros((batch_size, 1), device=device)
        gen_loss = F.binary_cross_entropy_with_logits(d_fake, real_labels)
        gen_acc = (torch.sigmoid(d_fake) < 0.5).float().mean().item()
        self.discriminator.train()
        return gen_loss, gen_acc

    def step_discriminator(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> tuple[torch.Tensor, float, float]:
        if self.cfg.loss_coefficient_disc > 0:
            return self._step_discriminator(real_data, fake_data)
        else:
            return torch.tensor(0.0), 0.0, 0.0
    
    def step_generator(self, fake_data: torch.Tensor) -> tuple[torch.Tensor, float]:
        if self.cfg.loss_coefficient_adv > 0:
            return self._step_generator(fake_data)
        else:
            return torch.tensor(0.0), 0.0