import torch
from torch.utils.data import Dataset, DataLoader

class EnWikiDataset(Dataset):
    def __init__(self, instances):
        self.input_ids = instances['input_ids']
        self.input_mask = instances['input_mask']
        self.segment_ids = instances['segment_ids']
        self.next_sentence_labels = instances['next_sentence_labels']
        self.mlm_pred_positions = instances['mlm_pred_positions']
        self.mlm_pred_labels = instances['mlm_pred_label_ids']
        self.mlm_weights = instances['mlm_weights']

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        input_ids = torch.tensor(self.input_ids[index], dtype=torch.long)
        input_mask = torch.tensor(self.input_mask[index], dtype=torch.long)
        segment_ids = torch.tensor(self.segment_ids[index], dtype=torch.long)
        next_sentence_labels = torch.tensor(self.next_sentence_labels[index], dtype=torch.long)
        mlm_pred_positions = torch.tensor(self.mlm_pred_positions[index], dtype=torch.long)
        mlm_pred_labels = torch.tensor(self.mlm_pred_labels[index], dtype=torch.long)
        mlm_weights = torch.tensor(self.mlm_weights[index], dtype=torch.float)
        return (input_ids, input_mask, segment_ids, next_sentence_labels,
                mlm_pred_positions, mlm_pred_labels, mlm_weights)

def get_loader(instances, config, shuffle=None):
    dataset = EnWikiDataset(instances)
    loader = DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=shuffle,
                        pin_memory=config.pin_memory, num_workers=config.num_workers)
    return loader