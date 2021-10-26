from typing import List, Tuple

import torch
from ctcdecode import CTCBeamDecoder

from hw_asr.text_encoder.char_text_encoder import CharTextEncoder


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str]):
        super().__init__(alphabet)
        self.ind2char = {
            0: self.EMPTY_TOK
        }
        for text in alphabet:
            self.ind2char[max(self.ind2char.keys()) + 1] = text
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def ctc_decode(self, inds: List[int]) -> str:
        # TODO: your code here
        s = ''
        for ind in inds:
            a = self.ind2char[int(ind)]
            if s == '' or a != s[-1]:
                s += a
        res = ''
        for i in s:
            if i != '^':
                res += i
        return res

    def ctc_beam_search(self, probs: torch.tensor, probs_length, beam_size: int = 100) -> List[Tuple[str, float]]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        hypos = []
        # TODO: your code here
        decoder = CTCBeamDecoder(list(self.ind2char.values()), beam_width=beam_size)
        beams, scores, _, lens = decoder.decode(probs.unsqueeze(0))
        for i in range(beam_size):
            hypos.append((self.ctc_decode(beams[0][i][:lens[0][i]]), scores[0][i].item()))

        return sorted(hypos, key=lambda x: x[1], reverse=True)