from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

def optimizer_and_scheduler(model, total_steps, config):
    no_decay = ['bias', 'LayerNorm.weight']
    decay_params, no_decay_params = [], []

    for name, param in model.named_parameters():
        if any(n in name for n in no_decay):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    update_parameters = [
        {'params': decay_params, 'weight_decay': config.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]

    optimizer = AdamW(update_parameters, lr=config.lr, betas=config.betas, eps=config.adamw_eps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps,
                                                num_training_steps=total_steps)

    return optimizer, scheduler