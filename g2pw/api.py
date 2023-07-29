import os
import json
import requests
import zipfile
from io import BytesIO
import shutil

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from pypinyin import pinyin
from pypinyin import Style
from g2pw.module import G2PW
from g2pw.dataset import prepare_data, TextDataset, get_phoneme_labels, get_char_phoneme_labels
from g2pw.utils import load_config


def predict(model, dataloader, device, labels):
    model.eval()

    all_preds = []
    all_confidences = []
    with torch.no_grad():
        generator = dataloader
        for data in generator:
            input_ids, token_type_ids, attention_mask, phoneme_mask, char_ids, position_ids = \
                [data[name].to(device) for name in
                 ('input_ids', 'token_type_ids', 'attention_mask', 'phoneme_mask', 'char_ids', 'position_ids')]
            print(input_ids.shape)
            probs = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                phoneme_mask=phoneme_mask,
                char_ids=char_ids,
                position_ids=position_ids
            )

            max_probs, preds = map(lambda x: x.cpu().tolist(), probs.max(dim=-1))
            all_preds += [labels[pred] for pred in preds]
            all_confidences += max_probs

    return all_preds, all_confidences


class G2PWConverter:
    def __init__(self, model_dir='/home/duser/tts/fastSpeech2-master/g2pw_models',  model_source=None,
                 use_cuda=True, num_workers=None, batch_size=None,
                 turnoff_tqdm=True, enable_non_tradional_chinese=False):
        # assert  os.path.exists(os.path.join(model_dir, 'best_accuracy.pth'))
        assert os.path.exists(os.path.join(model_dir, 'best_accuracy.pth'))

        self.config = load_config(os.path.join(model_dir, 'config.py'), use_default=True)
        self.num_workers = num_workers if num_workers else self.config.num_workers
        self.batch_size = batch_size if batch_size else self.config.batch_size
        self.model_source = model_source if model_source else self.config.model_source
        self.turnoff_tqdm = turnoff_tqdm
        self.enable_opencc = enable_non_tradional_chinese

        self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.tokenizer = BertTokenizer.from_pretrained(self.config.model_source)

        polyphonic_chars_path = os.path.join(model_dir, 'POLYPHONIC_CHARS.txt')
        monophonic_chars_path = os.path.join(model_dir, 'MONOPHONIC_CHARS.txt')
        self.polyphonic_chars = [line.split('\t') for line in open(polyphonic_chars_path).read().strip().split('\n')]
        self.monophonic_chars = [line.split('\t') for line in open(monophonic_chars_path).read().strip().split('\n')]
        self.labels, self.char2phonemes = get_char_phoneme_labels(
            self.polyphonic_chars) if self.config.use_char_phoneme else get_phoneme_labels(self.polyphonic_chars)

        self.chars = sorted(list(self.char2phonemes.keys()))
        self.pos_tags = TextDataset.POS_TAGS

        self.model = G2PW.from_pretrained(
            self.model_source,
            labels=self.labels,
            chars=self.chars,
            pos_tags=self.pos_tags,
            use_conditional=self.config.use_conditional,
            param_conditional=self.config.param_conditional,
            use_focal=self.config.use_focal,
            param_focal=self.config.param_focal,
            use_pos=self.config.use_pos,
            param_pos=self.config.param_pos
        )
        checkpoint = os.path.join(model_dir, 'best_accuracy.pth')
        self.model.load_state_dict(torch.load(checkpoint, map_location=self.device))
        self.model.to(self.device)

    def __call__(self, sentences):
        if isinstance(sentences, str):
            sentences = [sentences]
        print(sentences)
        import time
        t0 = time.time()
        texts, query_ids, sent_ids, partial_results = self._prepare_data(sentences)

        dataset = TextDataset(self.tokenizer, self.labels, self.char2phonemes, self.chars, texts, query_ids,
                              use_mask=self.config.use_mask, use_char_phoneme=self.config.use_char_phoneme,
                              window_size=self.config.window_size, for_train=False)

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            collate_fn=dataset.create_mini_batch,
            num_workers=self.num_workers
        )
        print("gen dataloader:", time.time() - t0)
        preds, confidences = predict(self.model, dataloader, self.device, self.labels)
        print("gen preds:", time.time() - t0)
        if self.config.use_char_phoneme:
            preds = [pred.split(' ')[1] for pred in preds]

        results = [partial_results]
        for sent_id, query_id, pred in zip(sent_ids, query_ids, preds):
            pred = pred.replace("u:", "v")
            results[sent_id][query_id] = pred
        results.append(query_ids)
        return results

    def _prepare_data(self, sentences):
        print(sentences)
        polyphonic_chars = set(self.chars)
        texts, query_ids, sent_ids = [], [], []
        for sent_id, sent in enumerate(sentences):
            partial_results = [item[0] for item in pinyin(sent, neutral_tone_with_five=True, style=Style.TONE3)]
            for i, char in enumerate(sent):
                if char in polyphonic_chars:
                    texts.append(sent)
                    query_ids.append(i)
                    sent_ids.append(sent_id)
        return texts, query_ids, sent_ids, partial_results