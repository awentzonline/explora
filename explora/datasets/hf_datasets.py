import datasets
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


def format_gpt4llm_row(row):
    instruction = row['instruction']
    if row['input']:
        instruction += '\n' + row['input']
    return dict(text=instruction + '\n' + row['output'])


DATASET_ROW_FORMATTERS = {
    'teknium/GPT4-LLM-Cleaned': format_gpt4llm_row,
}


class HfDatasetDataModule(pl.LightningDataModule):
    def __init__(
        self, dataset='wikitext:wikitext-2-raw-v1',
        tokenizer_name='bert-base-uncased', tokenizer=None,
        batch_size=10, shuffle=True,
        max_seq_len=256, num_workers=0, **kwargs
    ):
        super().__init__()
        dataset_parts = dataset.split(':')
        self.dataset_name = dataset_parts[0]
        if len(dataset_parts) > 1:
            self.dataset_config = dataset_parts[1]
        else:
            self.dataset_config = None

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.max_seq_length = max_seq_len
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(tokenizer_name)
        if dataset in DATASET_ROW_FORMATTERS:
            self.row_formatter = DATASET_ROW_FORMATTERS[dataset]
        else:
            self.row_formatter = None # lambda r: r['text']

    def setup(self, stage):
        self.dataset = datasets.load_dataset(self.dataset_name, self.dataset_config)

        for split in self.dataset.keys():
            if self.row_formatter:
                self.dataset[split] = self.dataset[split].map(
                    self.row_formatter,
                    batched=False
                )
            self.dataset[split] = self.dataset[split].filter(
                lambda x: len(x['text']) > 80
            )
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
            )
            self.dataset[split].set_format(type="torch", columns=['input_ids', 'attention_mask'])

    def train_dataloader(self):
        return DataLoader(
            self.dataset['train'], batch_size=self.batch_size,
            num_workers=self.num_workers, shuffle=self.shuffle,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset['validation'], batch_size=self.batch_size,
            num_workers=self.num_workers, shuffle=self.shuffle,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset['test'], batch_size=self.batch_size,
            num_workers=self.num_workers, shuffle=self.shuffle,
        )

    def convert_to_features(self, example_batch, indices=None):
        texts_or_text_pairs = example_batch['text']
        # Tokenize the text/text pairs
        features = self.tokenizer.batch_encode_plus(
            texts_or_text_pairs, max_length=self.max_seq_length,
            pad_to_max_length=True, truncation=True,
        )
        return features
