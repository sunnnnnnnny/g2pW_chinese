import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from g2pw.utils import tokenize_and_map

ANCHOR_CHAR = '▁'


def prepare_data(sent_path, lb_path=None):
    raw_texts = open(sent_path).read().rstrip().split('\n')
    query_ids = [raw.index(ANCHOR_CHAR) for raw in raw_texts]
    texts = [raw.replace(ANCHOR_CHAR, '') for raw in raw_texts]
    if lb_path is None:
        return texts, query_ids
    else:
        phonemes = open(lb_path).read().rstrip().split('\n')
        return texts, query_ids, phonemes


def get_phoneme_labels(polyphonic_chars):
    labels = sorted(list(set([phoneme for char, phoneme in polyphonic_chars])))
    char2phonemes = {}
    for char, phoneme in polyphonic_chars:
        if char not in char2phonemes:
            char2phonemes[char] = []
        char2phonemes[char].append(labels.index(phoneme))
    return labels, char2phonemes


def get_char_phoneme_labels(polyphonic_chars):
    labels = sorted(list(set([f'{char} {phoneme}' for char, phoneme in polyphonic_chars])))
    char2phonemes = {}
    for char, phoneme in polyphonic_chars:
        if char not in char2phonemes:
            char2phonemes[char] = []
        char2phonemes[char].append(labels.index(f'{char} {phoneme}'))
    return labels, char2phonemes


def prepare_pos(pos_path):
     return open(pos_path).read().rstrip().split('\n')


class TextDataset(Dataset):
    POS_TAGS = ['UNK', 'A', 'C', 'D', 'I', 'N', 'P', 'T', 'V', 'DE', 'SHI']

    def __init__(self, tokenizer, labels, char2phonemes, chars, texts, query_ids, phonemes=None, pos_tags=None,
                 use_mask=False, use_char_phoneme=False, use_pos=False, window_size=None, max_len=512, for_train=True):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.window_size = window_size
        self.for_train = for_train

        self.labels = labels
        self.char2phonemes = char2phonemes
        self.chars = chars
        self.texts = texts
        self.query_ids = query_ids
        self.phonemes = phonemes
        self.pos_tags = pos_tags

        self.use_mask = use_mask
        self.use_char_phoneme = use_char_phoneme
        self.use_pos = use_pos

        if window_size is not None:
            self.truncated_texts, self.truncated_query_ids = self._truncate_texts(self.window_size, texts, query_ids)

    def _truncate_texts(self, window_size, texts, query_ids):
        truncated_texts = []
        truncated_query_ids = []
        for text, query_id in zip(texts, query_ids):
            start = max(0, query_id - window_size // 2)
            end = min(len(text), query_id + window_size // 2)
            truncated_text = text[start:end]
            truncated_texts.append(truncated_text)

            truncated_query_id = query_id - start
            truncated_query_ids.append(truncated_query_id)
        return truncated_texts, truncated_query_ids

    def _truncate(self, max_len, text, query_id, tokens, text2token, token2text):
        truncate_len = max_len - 2
        if len(tokens) <= truncate_len:
            return (text, query_id, tokens, text2token, token2text)

        token_position = text2token[query_id]

        token_start = token_position - truncate_len // 2
        token_end = token_start + truncate_len
        font_exceed_dist = -token_start
        back_exceed_dist = token_end - len(tokens)
        if font_exceed_dist > 0:
            token_start += font_exceed_dist
            token_end += font_exceed_dist
        elif back_exceed_dist > 0:
            token_start -= back_exceed_dist
            token_end -= back_exceed_dist

        start = token2text[token_start][0]
        end = token2text[token_end - 1][1]

        return (
            text[start:end],
            query_id - start,
            tokens[token_start:token_end],
            [i - token_start if i is not None else None for i in text2token[start:end]],
            [(s - start, e - start) for s, e in token2text[token_start:token_end]]
        )

    def __getitem__(self, idx):
        text = (self.truncated_texts if self.window_size else self.texts)[idx].lower()
        query_id = (self.truncated_query_ids if self.window_size else self.query_ids)[idx]

        try:
            tokens, text2token, token2text = tokenize_and_map(self.tokenizer, text)
        except Exception:
            print(f'warning: text "{text}" is invalid')
            return self[(idx + 1) % len(self)]

        text, query_id, tokens, text2token, token2text = self._truncate(self.max_len, text, query_id, tokens, text2token, token2text)
        # import ipdb
        # ipdb.set_trace()
        # text 盛顿政府对威士忌酒暴乱的镇压得到了广泛的认同。
        # tokens ['盛', '顿', '政', '府', '对', '威', '士', '忌', '酒', '暴', '乱', '的', '镇', '压', '得', '到', '了', '广', '泛', '的', '认', '同', '。']
        # query_id 16 text[16] = '了'
        processed_tokens = ['[CLS]'] + tokens + ['[SEP]']

        input_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(processed_tokens))  # bert normal input
        token_type_ids = torch.tensor([0] * len(processed_tokens))  # bert normal input
        attention_mask = torch.tensor([1] * len(processed_tokens))  # bert normal input

        query_char = text[query_id]
        # query_char '了'
        # self.labels = [['a1', 'a4', 'a5', 'ai1', 'ai2', 'ai4', 'ao1', 'ao2', 'ao4', 'ba1', 'ba3', ...]
        # len(self.labels) = 650
        # len(self.char2phonemes) = 623
        # self.char2phonemes {'万': [518], '上': [454], '与': [583, 584], '丧': [444, 445]} len(self.char2phonemes)=623
        # self.chars = ['万', '上', '与', '丧', '中', '为', '丽', '么',...] len(self.chars) = 623
        # self.char2phonemes["了"] = [281, 296]

        phoneme_mask = [1 if i in self.char2phonemes[query_char] else 0 for i in range(len(self.labels))] \
            if self.use_mask else [1] * len(self.labels)
        # len(phoneme_mask) = 650 all 0
        # sum(phoneme_mask) = 2
        char_id = self.chars.index(query_char)
        # char_id 11
        position_id = text2token[query_id] + 1  # [CLS] token locate at first place
        # ...

        outputs = {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'phoneme_mask': phoneme_mask,
            'char_id': char_id,
            'position_id': position_id,
        }

        if self.use_pos and self.pos_tags is not None:
            # yes
            # len(self.pos_tags) = 79117
            # self.POS_TAGS ['UNK', 'A', 'C', 'D', 'I', 'N', 'P', 'T', 'V', 'DE', 'SHI']
            pos_id = self.POS_TAGS.index(self.pos_tags[idx])
            # pos_id 3
            outputs['pos_id'] = pos_id

        if self.for_train:
            # yes
            # len(self.phonemes) = 79117
            phoneme = self.phonemes[idx]
            # phoneme 'le5'
            # self.use_char_phoneme 0
            label_id = self.labels.index(f'{query_char} {phoneme}' if self.use_char_phoneme else phoneme)
            # label_id 281
            outputs['label_id'] = label_id

        info = {
            'text': text,
            'tokens': tokens,
            'text2token': text2token,
            'token2text': token2text
        }
        outputs['info'] = info
        return outputs

    def __len__(self):
        return len(self.texts)

    def create_mini_batch(self, samples):

        def _agg(name):
            return [sample[name] for sample in samples]

        # zero pad 到同一序列長度
        input_ids = pad_sequence(_agg('input_ids'), batch_first=True)
        token_type_ids = pad_sequence(_agg('token_type_ids'), batch_first=True)
        attention_mask = pad_sequence(_agg('attention_mask'), batch_first=True)
        phoneme_mask = torch.tensor(_agg('phoneme_mask'), dtype=torch.float)
        char_ids = torch.tensor(_agg('char_id'), dtype=torch.long)
        position_ids = torch.tensor(_agg('position_id'), dtype=torch.long)

        batch_output = {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'phoneme_mask': phoneme_mask,
            'char_ids': char_ids,
            'position_ids': position_ids
        }

        if self.use_pos and self.pos_tags is not None:
            pos_ids = torch.tensor(_agg('pos_id'), dtype=torch.long)
            batch_output['pos_ids'] = pos_ids

        if self.for_train:
            label_ids = torch.tensor(_agg('label_id'), dtype=torch.long)
            batch_output['label_ids'] = label_ids
        else:
            infos = _agg('info')
            batch_output['infos'] = infos

        return batch_output
