import csv
import logging
import os
from typing import Optional, Any

import torch
import torchaudio
from transformers import FastSpeech2ConformerTokenizer

from src.datasets.base_dataset import BaseDataset
from src.model.mel_spectrogram import MelSpectrogram, MelSpectrogramConfig
from src.utils.io_utils import read_json, write_json
from src.utils.consts import ROOT_PATH
from random import random
from tqdm.auto import tqdm
from torch import randint

logger = logging.getLogger(__name__)


class CustomDirDataset(BaseDataset):
    def __init__(self, dataset_dir: str = 'LJSpeech-1.1', part='train', max_text_length: int = 128, *args, **kwargs):
        self.dataset_dir = dataset_dir
        self.index_file = ROOT_PATH / self.dataset_dir / 'index.json'
        self.part = part
        if self.index_file.exists():
            index = read_json(str(self.index_file))
            self.create_texts()
        else:
            index = self.create_texts(create_index=True)
        super(CustomDirDataset, self).__init__(index, part, *args, **kwargs)

        mel_config = MelSpectrogramConfig()
        self.mel_spectrogram = MelSpectrogram(mel_config)
        self.text_tokenizer = FastSpeech2ConformerTokenizer.from_pretrained("espnet/fastspeech2_conformer")
        self.max_text_length = max_text_length

    def create_texts(self, create_index: bool = False) -> dict[str, list[Any]]:
        metadata_file = ROOT_PATH / self.dataset_dir / 'metadata.csv'
        if not metadata_file.exists():
            logger.error(f'There is no such file: {metadata_file}')
            raise FileNotFoundError()
        index = dict(train=[], val=[])
        texts_dir = ROOT_PATH / self.dataset_dir / 'texts'
        if not texts_dir.exists():
            os.mkdir(texts_dir)
        wavs_dir = ROOT_PATH / self.dataset_dir / 'wavs'
        with open(metadata_file, 'r') as rfile:
            reader = csv.reader(rfile, delimiter='|')
            for row in tqdm(reader):
                name, text = row[0], row[1]
                text_file = texts_dir / f'Utterance_{name[2:]}.txt'
                with open(text_file, 'w') as wfile:
                    wfile.write(text)
                if create_index:
                    item = dict(
                        audio=str(wavs_dir / f'{name}.wav'),
                        text=str(text_file)
                    )
                    rndm = random()
                    if rndm < 0.8:
                        index['train'].append(item)
                    else:
                        index['val'].append(item)
        if len(index['train']) > 0 and len(index['val']) > 0:
            write_json(index, str(self.index_file))
            return index

    def __getitem__(self, item):
        data_dir = self._index[self.part][item]
        audio_path = data_dir['audio']
        audio, _ = torchaudio.load(audio_path)
        text = data_dir['text']
        wav_len = 8192
        audio_start = randint(low=0, high=audio.shape[1] - wav_len, size=(1,))
        audio = audio[:, audio_start: audio_start + wav_len]
        instance = dict(
            audio=audio,
            text=text,
            audio_path=audio_path,
            target_mel=self.mel_spectrogram(audio),
        )
        return instance
