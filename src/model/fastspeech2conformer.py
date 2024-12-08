from transformers import FastSpeech2ConformerTokenizer, FastSpeech2ConformerWithHifiGan
import soundfile as sf
from torch import nn
from src.utils.consts import ROOT_PATH


class TextToMelConverter(nn.Module):
    def __init__(self, converted_wav_dir: str = 'converted_data'):
        super(TextToMelConverter, self).__init__()
        self.translated_wav_dir = ROOT_PATH / converted_wav_dir
        self.tokenizer = FastSpeech2ConformerTokenizer.from_pretrained("espnet/fastspeech2_conformer")
        self.model = FastSpeech2ConformerWithHifiGan.from_pretrained("espnet/fastspeech2_conformer_with_hifigan")

    def create_wav(self, text, audio_id):
        inputs = self.tokenizer("Hello, my dog is cute.", return_tensors="pt")
        input_ids = inputs["input_ids"]

        output_dict = self.model(input_ids, return_dict=True)
        waveform = output_dict["waveform"]

        sf.write("speech.wav", waveform.squeeze().detach().numpy(), samplerate=22050)