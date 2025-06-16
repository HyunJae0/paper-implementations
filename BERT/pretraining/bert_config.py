import torch

class PreTrainingBertConfig:
    def __init__(
            self,
            tokenizer_path='bert-base-uncased',
            vocab_size=30522,
            hidden_size=768,
            intermediate_size=3072,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            num_hidden_layers=12,
            num_attention_heads=12,
            max_seq_len=128,
            short_seq_prob=0.1,
            num_duplication = 10,
            whole_word_mask=True,
            mlm_prob=0.15,
            max_mlm_pred_per_sequence=20,
            output_filename='en_wiki_instances.jsonl',
            batch_size=32,
            num_workers=4,
            pin_memory=True,
            initializer_range = 0.02,
            lr = 1e-4,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            adamw_eps = 1e-6,
            warmup_steps = 100,
            clip = 0.1,
            epochs=2,
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            ):

        # model params
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # mlm, nsp params
        self.tokenizer_path = tokenizer_path
        self.max_seq_len = max_seq_len
        self.short_seq_prob = short_seq_prob
        self.num_duplication = num_duplication
        self.whole_word_mask = whole_word_mask
        self.mlm_prob = mlm_prob
        self.max_mlm_pred_per_sequence = max_mlm_pred_per_sequence

        # dataloader params
        self.output_filename = output_filename
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # optimizer, train params
        self.initializer_range = initializer_range
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.adamw_eps = adamw_eps
        self.epochs = epochs
        self.clip = clip
        self.device = device