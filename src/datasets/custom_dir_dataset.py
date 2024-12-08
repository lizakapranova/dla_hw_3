import csv
import logging
from typing import Optional, Any

import torchaudio
from transformers import FastSpeech2ConformerTokenizer

from src.datasets.base_dataset import BaseDataset
from src.model.mel_spectrogram import MelSpectrogram, MelSpectrogramConfig
from src.utils.io_utils import read_json, write_json
from src.utils.consts import ROOT_PATH
from random import random
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class CustomDirDataset(BaseDataset):
    def __init__(self, dataset_dir: str = 'LJSpeech-1.1', part='train', mel_config: Optional[MelSpectrogramConfig] = None, max_text_length: int = 128, *args, **kwargs):
        self.dataset_dir = dataset_dir
        self.index_file = ROOT_PATH / self.dataset_dir / 'index.json'
        self.part = part
        if self.index_file.exists():
            index = read_json(str(self.index_file))
            self.create_texts()
        else:
            index = self.create_texts(create_index=True)
        super(CustomDirDataset, self).__init__(index, part, *args, **kwargs)

        mel_config = mel_config or MelSpectrogramConfig()
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
        wavs_dir = ROOT_PATH / self.dataset_dir / 'wavs'
        with open(metadata_file, 'r') as rfile:
            reader = csv.reader(rfile, delimiter='0x7c')
            for name, text in tqdm(reader):
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
        if index:
            write_json(index, str(self.index_file))
            return index

    def __getitem__(self, item):
        data_dir = self._index[self.part][item]
        audio_path = data_dir['audio']
        audio = self.load_audio(audio_path)
        text = data_dir['text']
        tokenized_text = self.tokenize_text(text)
        spectrogram = self.get_spectrogram(audio)
        instance = dict(
            audio=audio,
            text=text,
            audio_path=audio_path,
            target_mel=spectrogram,
            tokenized_text=tokenized_text
        )
        return instance

    @staticmethod
    def load_audio(audio_path):
        return torchaudio.load(audio_path)

    def get_spectrogram(self, audio):
        spec = self.mel_spectrogram(audio)
        return spec

    def tokenize_text(self, text):
        inputs = self.text_tokenizer(text, return_tensors="pt", max_length=self.max_text_length, padding='max_length')
        return inputs
