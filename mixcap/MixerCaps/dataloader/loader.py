from .dataset import *
from config import cfg

num_worker = 48

class Loader:
    def __init__(self, dataset, idx):
        self.dataset = dataset
        self.train_idx = idx[0: int(round(0.8 * len(idx)))]
        self.test_idx = idx[int(round(0.8 * len(idx))): len(idx)]

    def get(self):
        if self.dataset == 'LIVE':
            live_train_idx = self.train_idx
            live_test_idx = self.test_idx

            live_train_transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomCrop(size=cfg.DATASET.PATCH_SIZE),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))
            ])
            live_test_transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(size=cfg.DATASET.PATCH_SIZE),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))
            ])


            live_train_data = LIVEFolder(root=cfg.DATASET.LIVE.PATH, index=live_train_idx, transform=live_train_transforms,
                                         patch_num=cfg.DATASET.AUGMENTATION)
            live_test_data = LIVEFolder(root=cfg.DATASET.LIVE.PATH, index=live_test_idx, transform=live_test_transforms,
                                        patch_num=cfg.DATASET.AUGMENTATION)

            live_train_loader = torch.utils.data.DataLoader(live_train_data, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=True,
                                                            num_workers=num_worker)
            live_test_loader = torch.utils.data.DataLoader(live_test_data, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=False,
                                                           num_workers=num_worker)

            return live_train_loader, live_test_loader

        if self.dataset == 'LIVEC':
            livec_train_index = self.train_idx
            livec_test_index = self.test_idx

            train_transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomCrop(size=cfg.DATASET.PATCH_SIZE),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))
            ])
            test_transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(size=cfg.DATASET.PATCH_SIZE),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))
            ])

            livec_train_data = LIVEChallengeFolder(root=cfg.DATASET.LIVEC.PATH, index=livec_train_index, transform=train_transforms,
                                                   patch_num=cfg.DATASET.AUGMENTATION)
            livec_test_data = LIVEChallengeFolder(root=cfg.DATASET.LIVEC.PATH, index=livec_test_index, transform=test_transforms,
                                                  patch_num=cfg.DATASET.AUGMENTATION)

            livec_train_loader = torch.utils.data.DataLoader(livec_train_data, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=True,
                                                             num_workers=num_worker)
            livec_test_loader = torch.utils.data.DataLoader(livec_test_data, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=False,
                                                            num_workers=num_worker)

            return livec_train_loader, livec_test_loader
        
        if self.dataset == 'KonIQ-10K':
            koniq_train_idx = self.train_idx
            koniq_test_idx = self.test_idx
            koniq_10k_train_transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.Resize((512, 384)),
                torchvision.transforms.RandomCrop(size=cfg.DATASET.PATCH_SIZE),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

            koniq_10k_test_transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize((512, 384)),
                torchvision.transforms.RandomCrop(size=cfg.DATASET.PATCH_SIZE),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])


            koniq_train_data = Koniq_10kFolder(root=cfg.DATASET.KONIQ.PATH, index=koniq_train_idx,
                                               transform=koniq_10k_train_transforms, patch_num=cfg.DATASET.AUGMENTATION)
            koniq_test_data = Koniq_10kFolder(root=cfg.DATASET.KONIQ.PATH, index=koniq_test_idx,
                                              transform=koniq_10k_test_transforms, patch_num=cfg.DATASET.AUGMENTATION)

            koniq_train_loader = torch.utils.data.DataLoader(koniq_train_data, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=True,
                                                             num_workers=num_worker)
            koniq_test_loader = torch.utils.data.DataLoader(koniq_test_data, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=False,
                                                            num_workers=num_worker)

            return koniq_train_loader, koniq_test_loader

        if self.dataset == 'SPAQ':
            spaq_train_idx = self.train_idx
            spaq_test_idx = self.test_idx

            spaq_train_transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(),
                # torchvision.transforms.Resize((512, 512)),
                torchvision.transforms.RandomCrop(size=cfg.DATASET.PATCH_SIZE),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

            spaq_test_transforms = torchvision.transforms.Compose([
                # torchvision.transforms.Resize((512, 512)),
                torchvision.transforms.RandomCrop(size=cfg.DATASET.PATCH_SIZE),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

            spaq_train_data = SPAQ_Folder(root=cfg.DATASET.SPAQ.PATH, index=spaq_train_idx,
                                               transform=spaq_train_transforms, patch_num=cfg.DATASET.AUGMENTATION)
            spaq_test_data = SPAQ_Folder(root=cfg.DATASET.SPAQ.PATH, index=spaq_test_idx,
                                              transform=spaq_test_transforms, patch_num=cfg.DATASET.AUGMENTATION)

            spaq_train_loader = torch.utils.data.DataLoader(spaq_train_data, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=True,
                                                             num_workers=num_worker)
            spaq_test_loader = torch.utils.data.DataLoader(spaq_test_data, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=False,
                                                            num_workers=num_worker)

            return spaq_train_loader, spaq_test_loader

        if self.dataset == 'TID2013':
            tid_2013_train_idx = self.train_idx
            tid_2013_test_idx = self.test_idx

            tid_2013_train_transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomCrop(size=cfg.DATASET.PATCH_SIZE),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))
            ])
            tid_2013_test_transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(size=cfg.DATASET.PATCH_SIZE),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))
            ])


            tid_2013_train_data = TID2013Folder(root=cfg.DATASET.TID2013.PATH, index=tid_2013_train_idx,
                                                transform=tid_2013_train_transforms, patch_num=cfg.DATASET.AUGMENTATION)
            tid_2013_test_data = TID2013Folder(root=cfg.DATASET.TID2013.PATH, index=tid_2013_test_idx,
                                               transform=tid_2013_test_transforms, patch_num=cfg.DATASET.AUGMENTATION)


            tid_2013_train_loader = torch.utils.data.DataLoader(tid_2013_train_data, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=True,
                                                                num_workers=num_worker)
            tid_2013_test_loader = torch.utils.data.DataLoader(tid_2013_test_data, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=False,
                                                               num_workers=num_worker)

            return tid_2013_train_loader, tid_2013_test_loader
