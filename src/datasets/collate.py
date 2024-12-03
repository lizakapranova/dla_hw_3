import torch
from torch.nn.utils.rnn import pad_sequence

SPECTROGRAM_PADDING = -11.5129251

def pad_spectrogram(spectrograms, max_length):
    padded_specs = []
    for spectrogram in spectrograms:
        padding = torch.ones((spectrogram.shape[0], spectrogram.shape[1], max_length - spectrogram.shape[2]))
        padding *= SPECTROGRAM_PADDING
        padded_spec = torch.concat([spectrogram, padding], dim=2)
        padded_specs.append(padded_spec)
    return padded_specs



def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {}

    sorted_items = sorted(dataset_items, key=lambda item: item['spectrogram'].shape[2], reverse=True)

    max_spectrogram_length = sorted_items[0]['spectrogram'].shape[2]

    result_batch['audio_path'] = [x['audio_path'] for x in dataset_items]
    result_batch["audio"] = pad_sequence([x["audio"].squeeze(0) for x in dataset_items]).permute(1, 0)
    result_batch['text'] = [x['text'] for x in dataset_items]
    result_batch['spectrogram'] = pad_spectrogram([x['spectrogram'] for x in dataset_items], max_spectrogram_length)

    return result_batch
