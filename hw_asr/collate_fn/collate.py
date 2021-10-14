import logging
from typing import List
from hw_asr.text_encoder.char_text_encoder import CharTextEncoder
from torch import tensor, cat, transpose
from torch.nn.functional import pad

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    result_batch = {
        'text': [],
        'text_encoded': [],
        'text_encoded_length': [],
        'spectrogram': [],
        'spectrogram_length': []
    }
    # TODO: your code here

    for item in dataset_items:
        result_batch['spectrogram_length'].append(item['spectrogram'].size()[-1])
        result_batch['text_encoded_length'].append(item['text_encoded'].size()[-1])
        result_batch['text'].append(CharTextEncoder.normalize_text(item['text']))

    result_batch['text_encoded_length'] = tensor(result_batch['text_encoded_length'])
    result_batch['spectrogram_length'] = tensor(result_batch['spectrogram_length'])

    max_sp_length = result_batch['spectrogram_length'].max().item()
    max_txt_length = result_batch['text_encoded_length'].max().item()

    for item in dataset_items:
        result_batch['spectrogram'].append(
            pad(item['spectrogram'], (0, max_sp_length - item['spectrogram'].shape[-1])))
        result_batch['text_encoded'].append(
            pad(item['text_encoded'], (0, max_txt_length - item['text_encoded'].shape[-1])))

    result_batch['text_encoded'] = cat(result_batch['text_encoded'])
    result_batch['spectrogram'] = cat(result_batch['spectrogram']).transpose(1, 2)

    return result_batch

