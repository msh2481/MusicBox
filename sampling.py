import math

import soundfile
import torch
import torch.nn.functional as F
from tqdm import tqdm


def beam_search(
    model,
    beam_size,
    branch_factor,
    sample_length,
    temperature,
    seed,
    logging=None,
):
    alphabet, seed_length = seed.size()
    beam = seed.repeat(beam_size, 1, 1)
    assert beam.shape == (beam_size, *seed.shape)
    nll = torch.zeros(beam_size)

    for iter in tqdm(range(sample_length - seed.size(1))):
        logits = model(beam)[:, :, -1]
        assert logits.shape == (beam_size, alphabet)
        tokens = torch.multinomial(
            torch.softmax(logits / temperature, dim=-1), branch_factor
        )
        logits = F.log_softmax(logits, dim=-1).gather(-1, tokens)
        tokens, logits = tokens.flatten(), logits.flatten()
        where = torch.arange(beam_size).repeat_interleave(branch_factor)
        assert tokens.shape == (beam_size * branch_factor,)
        assert logits.shape == (beam_size * branch_factor,)
        assert where.shape == (beam_size * branch_factor,)

        srt = torch.argsort(logits, descending=True)[:beam_size]
        one_hot = F.one_hot(tokens[srt], alphabet)
        assert one_hot.shape == (beam_size, alphabet)
        beam = torch.cat((beam[where[srt]], one_hot.unsqueeze(2)), dim=2)
        nll = nll[where[srt]] - logits[srt]

        if logging:
            idx = nll.argmin()
            logging(beam[idx], nll[idx])
    idx = nll.argmin()
    return beam[idx], nll[idx].item()