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
            real_data=real_data,
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

        disc_loss = -torch.mean(torch.log(d_real_rel) + torch.log(1 - d_fake_rel))
        disc_acc_real = (d_real_logits > avg_fake_logits).float().mean().item()
        disc_acc_fake = (d_fake_logits < avg_real_logits).float().mean().item()

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

        avg_real_logits = d_real_logits.mean()
        avg_fake_logits = d_fake_logits.mean()

        d_fake_rel = torch.sigmoid(d_fake_logits - avg_real_logits)
        d_real_rel = torch.sigmoid(d_real_logits - avg_fake_logits)

        gen_loss = -torch.mean(torch.log(d_fake_rel) + torch.log(1 - d_real_rel))
        gen_acc = (d_fake_logits > avg_real_logits).float().mean().item()

        self.discriminator.train()
        return gen_loss, gen_acc