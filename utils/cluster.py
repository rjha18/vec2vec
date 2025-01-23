from argparse import Namespace

import datasets


class CdeDatasetWrapper(datasets.Dataset):
    dataset: datasets.Dataset
    def __init__(
            self, 
            dataset: datasets.Dataset,
        ):
        from cde.dataset import get_subdomain_idxs_cached
        self.dataset = dataset
        self.subdomain_idxs = get_subdomain_idxs_cached(self.dataset)
        self._fingerprint = dataset._fingerprint

    
    def __len__(self):
        return len(self.dataset)


def make_cluster_sampler(dset: datasets.Dataset, cfg: Namespace):
    from cde.sampler import AutoClusterWithinDomainSampler

    dset_wrapper = CdeDatasetWrapper(dset)
    return AutoClusterWithinDomainSampler(
        dataset=dset_wrapper,
        query_to_doc=True,
        batch_size=2048,
        cluster_size=cfg.bs,
        shuffle=True,
        share_negatives_between_gpus=False,
        downscale_and_normalize=True,
        # model="gtr_base",
        model="bert",
        seed=cfg.seed,
    )
