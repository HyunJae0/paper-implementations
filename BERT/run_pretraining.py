import json
import random
import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm.auto import tqdm
from get_pretraining_data import *
from bert_pretraining import *


def init_weights(module, initializer_range=0.02):
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=initializer_range)
    elif isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, std=initializer_range)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    config = PreTrainingBertConfig()
    tokenizer = HuggingFaceTokenizer(config.tokenizer_path)

    print('Loading dataset...')
    tokenizer = HuggingFaceTokenizer(config.tokenizer_path)
    wiki = load_dataset("lucadiliello/english_wikipedia", split='train')
    wiki = wiki.remove_columns(['filename', 'source_domain', 'title', 'url'])

    print('Preprocessing documents...')
    processed_wiki = wiki.map(process_batch, batched=True,
                              batch_size=1000, num_proc=4,
                              remove_columns=wiki.column_names)

    print('Tokenizing documents...')
    tokenization_wiki = processed_wiki.map(tokenize_batch, batched=True,
                                           batch_size=1000, num_proc=4,
                                           remove_columns=processed_wiki.column_names)

    print('Creating augmented instances (MLM & NSP)...')
    all_mlm_token_sequences, all_mlm_pred_positions, all_mlm_pred_labels, \
        all_segment_ids, all_nsp_labels = create_augmented_instances(
        tokenizer, tokenization_wiki, config)

    print(f'Padding and saving instances to {config.output_filename}...')
    add_padding_and_save_instances(
        all_mlm_token_sequences, all_mlm_pred_positions,
        all_mlm_pred_labels, all_segment_ids, all_nsp_labels,
        tokenizer, config)

    with open(config.output_filename, 'r') as f:
        instances = json.load(f)

    train_loader = get_loader(instances, config, shuffle=True)
    print('Data Preparation Complete')

    print('Training Phase...')
    model = PreTrainingBertModel(config).to(config.device)
    model.apply(init_weights)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    criterion = nn.CrossEntropyLoss().to(config.device)
    total_steps = len(train_loader) * config.epochs
    optimizer, scheduler = optimizer_and_scheduler(model, total_steps, config)

    for epoch in range(config.epochs):
        train_loss, train_mlm_loss, train_nsp_loss = train(model, train_loader, criterion, optimizer, scheduler, config)
        print(f'Train Loss: {train_loss:.4f} | MLM Loss: {train_mlm_loss:.4f} | NSP Loss: {train_nsp_loss:.4f}')

    torch.save(model.state_dict(), 'pretraining_model.pt')
    print('Training Complete...')
