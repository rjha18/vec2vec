from typing import Dict, Iterable, Union
import wandb
import torch

from .dist import gather, get_rank

class TensorRunningAverages:
    _store_sum: Dict[str, torch.Tensor]
    _store_total: Dict[str, torch.Tensor]

    def __init__(self):
        self._store_sum = {}
        self._store_total = {}
    
    def __iter__(self) -> Iterable[str]:
        return iter(self._store_sum.keys())

    def update(self, key: str, val: Union[int, float, torch.Tensor]) -> None:
        if key not in self._store_sum:
            self.clear(key)
        if not isinstance(val, torch.Tensor):
            val = torch.tensor(val) # tensor -> num
        val = val.cpu()
        self._store_sum[key] += val
        self._store_total[key] += 1

    def get(self, key: str) -> float:
        total = max(self._store_total.get(key), torch.tensor(1.0))
        return (self._store_sum[key] / float(total.item())) or 0.0
    
    def clear(self, key: str) -> None:
        self._store_sum[key] = torch.tensor(0.0, dtype=torch.float32)
        self._store_total[key] = torch.tensor(0, dtype=torch.int32)
    
    def clear_all(self) -> None:
        for key in self._store_sum:
            self.clear(key)

    def get_and_clear_all(self) -> Dict[str, float]:
        metrics = {}
        for key in self:
            metrics[key] = self.get(key)
            self.clear(key)
        return metrics

class Logger:
    def __init__(self, **kws):
        self.vals = TensorRunningAverages()
        self.enabled = (not kws.pop("dummy", False)) and (get_rank() == 0)
        if self.enabled:
            wandb.init(
                **kws
            )
        else:
            print("Wandb is disabled")
            wandb.init(mode="disabled")
        self.log_frequency = 250
        self.log_step = 0

    def logkv(self, k, v):
        val = v.detach() if isinstance(v, torch.Tensor) else torch.tensor(v)
        self.vals.update(k, val)
        return v
    
    def dumpkvs(self, force=False):
        self.log_step += 1
        if self.log_step % self.log_frequency == 0 or force:
            metrics = self.vals.get_and_clear_all()
            # metrics = {k: gather(v.cuda()).mean().cpu() for k, v in metrics.items()}
            if self.enabled:
                wandb.log(metrics)
