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

    def _step_generator(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> tuple[torch.Tensor, float]:
        self.discriminator.eval()
        d_fake_logits = self.discriminator(fake_data)
        device = fake_data.device
        batch_size = fake_data.size(0)
        real_labels = torch.zeros((batch_size, 1), device=device)
        gen_loss = F.binary_cross_entropy_with_logits(d_fake_logits, real_labels)
        gen_acc = (d_fake_logits.sigmoid() < 0.5).float().mean().item()
        self.discriminator.train()
        return gen_loss, gen_acc

    def step_discriminator(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> tuple[torch.Tensor, float, float]:
        if self.cfg.loss_coefficient_disc > 0:
            return self._step_discriminator(real_data, fake_data)
        else:
            return torch.tensor(0.0), 0.0, 0.0

    def step_generator(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> tuple[torch.Tensor, float]:
        if self.cfg.loss_coefficient_adv > 0:
            return self._step_generator(real_data=real_data, fake_data=fake_data)
        else:
            return torch.tensor(0.0), 0.0

    def step(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> tuple[
            torch.Tensor, torch.Tensor, float, float, float]:
        disc_loss, disc_acc_real, disc_acc_fake = self.step_discriminator(
            real_data=real_data.detach(),
            fake_data=fake_data.detach()
        )
        gen_loss, gen_acc = self.step_generator(
            real_data=real_data,
            fake_data=fake_data
        )
        return disc_loss, gen_loss, disc_acc_real, disc_acc_fake, gen_acc


class RelativisticGAN(VanillaGAN):
    def _step_discriminator(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> tuple[torch.Tensor, float, float]:
        self.generator.eval()
        
        d_real_logits = self.discriminator(real_data)
        d_fake_logits = self.discriminator(fake_data)
        
        avg_fake_logits = d_fake_logits.mean()
        avg_real_logits = d_real_logits.mean()
        
        d_real_rel = torch.sigmoid(d_real_logits - avg_fake_logits)
        d_fake_rel = torch.sigmoid(d_fake_logits - avg_real_logits)
        
        eps = 1e-7
        disc_loss = -torch.mean(torch.log(d_real_rel + eps) + torch.log(1 - d_fake_rel + eps))
        
        disc_acc_real = (d_real_logits > avg_fake_logits).float().mean().item()
        disc_acc_fake = (d_fake_logits < avg_real_logits).float().mean().item()
        
        self.generator.train()
        self.discriminator_opt.zero_grad()
        self.accelerator.backward(disc_loss * self.cfg.loss_coefficient_disc)
        
        self.accelerator.clip_grad_norm_(self.discriminator.parameters(), self.cfg.max_grad_norm)
        self.discriminator_opt.step()
        
        return disc_loss, disc_acc_real, disc_acc_fake

    def _step_generator(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> tuple[torch.Tensor, float]:
        self.discriminator.eval()
        
        d_real_logits = self.discriminator(real_data)
        d_fake_logits = self.discriminator(fake_data)
        
        avg_real_logits = d_real_logits.mean()
        avg_fake_logits = d_fake_logits.mean()
        
        d_fake_rel = torch.sigmoid(d_fake_logits - avg_real_logits)
        d_real_rel = torch.sigmoid(d_real_logits - avg_fake_logits)
        
        # Corrected generator loss
        eps = 1e-7
        gen_loss = -torch.mean(torch.log(1 - d_real_rel + eps) + torch.log(d_fake_rel + eps))
        gen_acc = (d_fake_logits > avg_real_logits).float().mean().item()
        
        self.discriminator.train()
        return gen_loss, gen_acc


class WassersteinGAN(VanillaGAN):
    def compute_gradient_penalty(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> torch.Tensor:
        batch_size = real_data.size(0)
        alpha = torch.rand_like(real_data)
        # Interpolate between real and fake data
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)
        
        # Calculate critic scores
        critic_interpolated = self.discriminator(interpolated)
        
        # Calculate gradients of scores with respect to interpolated inputs
        gradients = torch.autograd.grad(
            outputs=critic_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(critic_interpolated),
            create_graph=True,
            retain_graph=True,
        )[0]
        
        # Calculate gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
        return gradient_penalty

    def _step_discriminator(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> tuple[torch.Tensor, float, float]:
        self.generator.eval()

        # Calculate critic scores
        d_real_logits = self.discriminator(real_data)
        d_fake_logits = self.discriminator(fake_data)
        
        # Wasserstein loss
        disc_loss = d_fake_logits.mean() - d_real_logits.mean()
        
        # Gradient penalty
        gp = self.compute_gradient_penalty(real_data, fake_data)
        lambda_gp = 10  # Gradient penalty coefficient
        
        # Total critic loss
        disc_loss = disc_loss + lambda_gp * gp
        
        # Calculate accuracy (optional, for monitoring only)
        disc_acc_real = (d_real_logits > 0).float().mean().item()
        disc_acc_fake = (d_fake_logits < 0).float().mean().item()
        
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
        d_fake_logits = self.discriminator(fake_data)
        
        # Wasserstein loss for generator (minimize negative critic score)
        gen_loss = -d_fake_logits.mean()
        
        # Calculate accuracy (optional, for monitoring only)
        gen_acc = (d_fake_logits > 0).float().mean().item()
        
        self.discriminator.train()
        return gen_loss, gen_acc