import csv
import logging
from typing import Optional

import torchaudio

from src.datasets.base_dataset import BaseDataset
from src.model.mel_spectrogram import MelSpectrogram, MelSpectrogramConfig
from src.utils.io_utils import ROOT_PATH, read_json

logger = logging.getLogger(__name__)


class CustomDirDataset(BaseDataset):
    def __init__(self, dataset_dir: str = 'LJSpeech-1.1', mel_config: Optional[MelSpectrogramConfig] = None):
        self.dataset_dir = dataset_dir
        self.index_file = ROOT_PATH / self.dataset_dir / 'index.json'
        if self.index_file.exists():
            index = read_json(str(self.index_file))
            self.create_texts()
        else:
            index = self.create_texts(create_index=True)
        super(CustomDirDataset, self).__init__(index)

        mel_config = mel_config or MelSpectrogramConfig()
        self.mel_spectrogram = MelSpectrogram(mel_config)

    def create_texts(self, create_index: bool = False) -> Optional[list[dict[str, str]]]:
        metadata_file = ROOT_PATH / self.dataset_dir / 'metadata.csv'
        if not metadata_file.exists():
            logger.error(f'There is no such file: {metadata_file}')
            raise FileNotFoundError()
        index = []
        texts_dir = ROOT_PATH / self.dataset_dir / 'texts'
        wavs_dir = ROOT_PATH / self.dataset_dir / 'wavs'
        with open(metadata_file, 'r') as rfile:
            reader = csv.reader(rfile, delimiter='0x7c')
            for name, text in reader:
                text_file = texts_dir / f'Utterance_{name[2:]}.txt'
                with open(text_file, 'w') as wfile:
                    wfile.write(text)
                if create_index:
                    item = dict(
                        audio=str(wavs_dir / f'{name}.wav'),
                        text=str(text_file)
                    )
                    index.append(item)
        if index:
            return index

    def __getitem__(self, item):
        data_dir = self._index[item]
        audio_path = data_dir['audio']
        audio = self.load_audio(audio_path)
        text = data_dir['text']
        spectrogram = self.get_spectrogram(audio)
        instance = dict(
            audio=audio,
            text=text,
            audio_path=audio_path,
            spectrogram=spectrogram,
        )
        return instance

    @staticmethod
    def load_audio(audio_path):
        return torchaudio.load(audio_path)

    def get_spectrogram(self, audio):
        spec = self.mel_spectrogram(audio)
        return spec
