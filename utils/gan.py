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
    discriminator_opt: torch.optim.Optimizer
    discriminator_scheduler: torch.optim.lr_scheduler._LRScheduler
    accelerator: accelerate.Accelerator

    @property
    def _batch_size(self) -> int:
        return self.cfg.bs
    
    def compute_gradient_penalty(self, real_data: torch.Tensor) -> torch.Tensor:
        real_data.requires_grad_(True)
        d_out = self.discriminator(real_data)
        
        gradients = torch.autograd.grad(
            outputs=d_out.sum(),
            inputs=real_data,
            create_graph=True,
            retain_graph=True,
        )[0]
        
        return gradients.pow(2).sum().mean()

    def _step_discriminator(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        d_real_logits, d_fake_logits = self.discriminator(real_data), self.discriminator(fake_data)

        device = d_real_logits.device
        batch_size = d_real_logits.size(0)
        real_labels = torch.ones((batch_size, 1), device=device) * (1 - self.cfg.smooth)
        fake_labels = torch.ones((batch_size, 1), device=device) * self.cfg.smooth
        disc_loss_real = F.binary_cross_entropy_with_logits(d_real_logits, real_labels)
        disc_loss_fake = F.binary_cross_entropy_with_logits(d_fake_logits, fake_labels)
        disc_loss = (disc_loss_real + disc_loss_fake) / 2
        disc_acc_real = (d_real_logits.sigmoid() < 0.5).float().mean().item()
        disc_acc_fake = (d_fake_logits.sigmoid() > 0.5).float().mean().item()

        r1_penalty = self.compute_gradient_penalty(real_data)

        self.generator.train()
        self.discriminator_opt.zero_grad()
        self.accelerator.backward(
            disc_loss + (r1_penalty * 1.0)  * self.cfg.loss_coefficient_disc
        )
        self.accelerator.clip_grad_norm_(
            self.discriminator.parameters(),
            self.cfg.max_grad_norm
        )
        self.discriminator_opt.step()
        self.discriminator_scheduler.step()
        return r1_penalty.detach(), disc_loss.detach(), disc_acc_real, disc_acc_fake

    def _step_generator(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> tuple[torch.Tensor, float]:
        d_fake_logits = self.discriminator(fake_data)
        device = fake_data.device
        batch_size = fake_data.size(0)
        real_labels = torch.zeros((batch_size, 1), device=device)
        gen_loss = F.binary_cross_entropy_with_logits(d_fake_logits, real_labels)
        gen_acc = (d_fake_logits.sigmoid() < 0.5).float().mean().item()
        return gen_loss, gen_acc

    def step_discriminator(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> tuple[torch.Tensor, float, float]:
        if self.cfg.loss_coefficient_disc > 0:
            return self._step_discriminator(real_data, fake_data)
        else:
            return torch.tensor(0.0), 0.0, 0.0

    def step_generator(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> tuple[torch.Tensor, float]:
        if self.cfg.loss_coefficient_gen > 0:
            return self._step_generator(real_data=real_data, fake_data=fake_data)
        else:
            return torch.tensor(0.0), 0.0

    def step(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, float, float, float]:
        self.generator.eval()
        self.discriminator.train()
        r1_penalty, disc_loss, disc_acc_real, disc_acc_fake = self.step_discriminator(
            real_data=real_data.detach(),
            fake_data=fake_data.detach()
        )
        self.generator.train()
        self.discriminator.eval()
        gen_loss, gen_acc = self.step_generator(
            real_data=real_data,
            fake_data=fake_data
        )

        return r1_penalty, disc_loss, gen_loss, disc_acc_real, disc_acc_fake, gen_acc


class RelativisticGAN(VanillaGAN):
    def _step_discriminator(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> tuple[torch.Tensor, float, float]:
        self.generator.eval()
        d_real_logits = self.discriminator(real_data)
        d_fake_logits = self.discriminator(fake_data)

        disc_loss = F.binary_cross_entropy_with_logits(d_fake_logits - d_real_logits, torch.ones_like(d_real_logits))
        disc_acc_real = (d_real_logits > d_fake_logits).float().mean().item()
        disc_acc_fake = 1.0 - disc_acc_real

        self.generator.train()
        self.discriminator_opt.zero_grad()
        self.accelerator.backward(disc_loss * self.cfg.loss_coefficient_disc)
        self.accelerator.clip_grad_norm_(
            self.discriminator.parameters(),
            self.cfg.max_grad_norm
        )
        self.discriminator_opt.step()

        return disc_loss, disc_acc_real, disc_acc_fake

    def _step_generator(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> tuple[torch.Tensor, float]:
        self.discriminator.eval()

        d_real_logits = self.discriminator(real_data)
        d_fake_logits = self.discriminator(fake_data)
        gen_loss = F.binary_cross_entropy_with_logits(d_real_logits - d_fake_logits, torch.ones_like(d_real_logits))

        gen_acc = (d_real_logits > d_fake_logits).float().mean().item()
        self.discriminator.train()
        return gen_loss, gen_acc
