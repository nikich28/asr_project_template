from typing import List, Tuple
from itertools import groupby


from torch import Tensor

from hw_asr.text_encoder.char_text_encoder import CharTextEncoder


class BPECharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "<PAD>"

    def __init__(self, alphabet, bpe_model):
        super().__init__(alphabet)
        self.bpe_model = bpe_model
        self.ind2char = {bpe_model.subword_to_id(i): i for i in alphabet}
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def encode(self, text) -> Tensor:
        text = self.normalize_text(text)
        try:
            return Tensor(self.bpe.encode(text)).unsqueeze(0)
        except KeyError as e:
            raise Exception(
                f"Can't encode text '{text}' with BPE'")

    @classmethod
    def get_simple_alphabet(cls, args):
        model = youtokentome.BPE(model=args['model_path'])
        return cls(alphabet=model.vocab(), model=model)

    def ctc_decode(self, inds: List[int]) -> str:
        inds = [int(x[0]) for x in groupby(inds)]
        res = self.model.decode(inds, ignore_ids=[0, 1, 2, 3])[0]
        return res