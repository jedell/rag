import logging
import json
import copy
from typing import Dict, Sequence, Tuple
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
import transformers
from retriever.nomic import encode_query

IGNORE_INDEX = -100

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    generator_tokenizer: transformers.PreTrainedTokenizer
    retriever_tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, retriever_inputs = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "retriever_inputs"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.generator_tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return (
            input_ids,
            labels,
            [retriever_input['input_ids'] for retriever_input in retriever_inputs],
            input_ids.ne(self.generator_tokenizer.pad_token_id), # attention mask
        )

def _tokenize_fn(
        strings: Sequence[str],
        generator_tokenizer: transformers.PreTrainedTokenizer,
    ) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        generator_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=generator_tokenizer.model_max_length,
            truncation=True,
            add_special_tokens=False
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(generator_tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

INST_START = "[INST]"
INST_END = "[/INST]"

def preprocess(
    samples: Dict,
    generator_tokenizer: transformers.PreTrainedTokenizer,
    retriever_tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Preprocess the data by tokenizing."""
    sources = []
    targets = []
    for sample in samples:
        for interaction in sample['interactions']:
            if interaction['is_user']:
                sources.append(interaction['text'])
        else:
            targets.append(interaction['text'])
    examples = [f"{INST_START} {s} {INST_END} {t}" for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, generator_tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX

    retriever_inputs = [
        encode_query(text, retriever_tokenizer)['input_ids'].squeeze(0)
        for text in sources
    ]
    context_src = [s['file'] for s in samples]

    return dict(input_ids=input_ids, labels=labels, retriever_inputs=retriever_inputs, context_src=context_src)

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self,
            data_path: str,
            generator_tokenizer: transformers.PreTrainedTokenizer,
            retriever_tokenizer: transformers.PreTrainedTokenizer
        ):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = self._jload(data_path)
        
        logging.warning("Formatting inputs...")
        sources = [
            example.get("user", "") for example in list_data_dict
        ]
        targets = [f"{example['assistant']}{generator_tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, generator_tokenizer, retriever_tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.retriever_inputs = data_dict["retriever_inputs"]
        self.labels = data_dict["labels"]

    def _jload(self, path):
        f = open(path, 'r')
        j = json.load(f)
        f.close()
        return j

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            retriever_inputs=self.retriever_inputs[i]
        )

def tokenize_instruct(
        sample: Dict,
        generator_tokenizer: transformers.PreTrainedTokenizer,
        retriever_tokenizer: transformers.PreTrainedTokenizer
    ) -> Dict:
        """Tokenize a list of strings."""
        tokens = []
        masks = []
        # TODO: handle if chain of user queries
        retriever_tokens = []
        eos_token_id = generator_tokenizer.eos_token_id

        for interaction in sample["interactions"]:
            msg = interaction["text"]
            if interaction["is_user"]:
                msg = f"{INST_START} {msg} {INST_END}"

                if not interaction["text"].startswith('search_query:'):
                    retriever_msg = f'search_query: {interaction["text"]}'
                else:
                    retriever_msg = interaction["text"]
                # TODO: handle if chain of user queries
                retriever_tokens.extend(retriever_tokenizer.encode(retriever_msg))

            curr_tokens = generator_tokenizer.encode(msg, add_special_tokens=False)

            is_bot = not interaction["is_user"]
            if is_bot: # is_bot
                curr_tokens.append(eos_token_id)
            mask = is_bot
            curr_masks = [mask] * len(curr_tokens)

            tokens.extend(curr_tokens)
            masks.extend(curr_masks)

        return dict(
            tokens=tokens,
            masks=masks,
            retriever_tokens=retriever_tokens,
            context_src=sample['file']
        )

@dataclass
class MistralCollator(object):
    """Collate examples for Mistral Instruct."""

    generator_pad_token_id: int
    retriever_pad_token_id: int

    def __call__(self, instances: Sequence[Dict]) -> Tuple:
        input_ids, labels, retriever_inputs = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels", "retriever_inputs")
        )

        # pad input_ids and labels
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.generator_pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
  
        retriever_inputs = torch.nn.utils.rnn.pad_sequence(
            retriever_inputs, batch_first=True, padding_value=self.retriever_pad_token_id
        )


        attn_mask = input_ids.ne(self.generator_pad_token_id)
        retriever_attn_mask = retriever_inputs.ne(self.retriever_pad_token_id)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attn_mask=attn_mask,
            retriever_tokens=retriever_inputs,
            retriever_attn_mask=retriever_attn_mask,
            context_src=[instance['context_src'] for instance in instances],
        )


class MistralInstructDataset(Dataset):
    """Dataset for Mistral Instruct."""

    def __init__(
            self,
            data_path: str,
            generator_tokenizer: transformers.PreTrainedTokenizer,
            retriever_tokenizer: transformers.PreTrainedTokenizer
        ):
        super(MistralInstructDataset, self).__init__()
        self.instruct_list = self._jload(data_path)

        samples = preprocess(self.instruct_list, generator_tokenizer, retriever_tokenizer)

        self.input_ids = samples["input_ids"]
        self.retriever_inputs = samples["retriever_inputs"]
        self.labels = samples["labels"]
        self.context_src = samples["context_src"]

    def _jload(self, path):
        f = open(path, 'r')
        j = json.load(f)
        f.close()
        return j

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            retriever_inputs=self.retriever_inputs[i],
            context_src=self.context_src[i]
        )

def build_mistral_instruct_dataset(
        data_path: str,
        generator_tokenizer: transformers.PreTrainedTokenizer,
        retriever_tokenizer: transformers.PreTrainedTokenizer,
    ) -> Dict:
    """Make dataset and collator for Mistral Instruct Formatted Dataset."""
    train_dataset = MistralInstructDataset(data_path, generator_tokenizer, retriever_tokenizer)
    data_collator = MistralCollator(
        generator_pad_token_id=generator_tokenizer.pad_token_id,
        retriever_pad_token_id=retriever_tokenizer.pad_token_id
    )
    return dict(train=train_dataset, eval=None, collator=data_collator)


def init_dataset(
        generator_tokenizer: transformers.PreTrainedTokenizer,
        retriever_tokenizer: transformers.PreTrainedTokenizer,
        data_path
    ) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(
        generator_tokenizer=generator_tokenizer,
        retriever_tokenizer=retriever_tokenizer,
        data_path=data_path
    )
    data_collator = DataCollatorForSupervisedDataset(
        generator_tokenizer=generator_tokenizer,
        retriever_tokenizer=retriever_tokenizer
    )
    return dict(train=train_dataset, eval=None, collator=data_collator)

