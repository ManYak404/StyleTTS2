import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import random
random.seed(0)

import numpy as np
np.random.seed(0)

import time
import yaml
from munch import Munch
import torchaudio
import soundfile as sf
import nltk
from nltk.tokenize import word_tokenize
import phonemizer
from datetime import datetime
import uuid

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading NLTK punkt_tab tokenizer...")
    nltk.download('punkt_tab', quiet=True)

from models import *
from utils import *
from text_utils import TextCleaner
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

# ----------------------------
# CONFIG
# ----------------------------
checkpoint_path = "~/voice-checkpoint/epoch_2nd_00049.pth"
config_path = "~/voice-checkpoint/config_ft.yml"
output_dir = "~/voice-checkpoint/outputs"
text = "Hello, this is a test of my trained model. I love baby seals because they are plump and cute!"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Generate unique filename with timestamp and UUID
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
unique_id = str(uuid.uuid4())[:8]
output_wav = os.path.join(output_dir, f"output_{timestamp}_{unique_id}.wav")

# ----------------------------
# SET DEVICE
# ----------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ----------------------------
# SETUP UTILITIES
# ----------------------------
to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

# Load phonemizer
global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True)
textclenaer = TextCleaner()

# ----------------------------
# LOAD CONFIG
# ----------------------------
print("Loading config...")
config = yaml.safe_load(open(config_path))

# ----------------------------
# LOAD PRETRAINED MODELS
# ----------------------------
print("Loading pretrained ASR model...")
ASR_config = config.get('ASR_config', False)
ASR_path = config.get('ASR_path', False)
text_aligner = load_ASR_models(ASR_path, ASR_config)

print("Loading pretrained F0 model...")
F0_path = config.get('F0_path', False)
pitch_extractor = load_F0_models(F0_path)

print("Loading BERT model...")
from Utils.PLBERT.util import load_plbert
BERT_path = config.get('PLBERT_dir', False)
plbert = load_plbert(BERT_path)

# ----------------------------
# BUILD MODEL
# ----------------------------
print("Building model...")
model = build_model(recursive_munch(config['model_params']), text_aligner, pitch_extractor, plbert)
_ = [model[key].eval() for key in model]
_ = [model[key].to(device) for key in model]

# ----------------------------
# LOAD CHECKPOINT
# ----------------------------
print(f"Loading checkpoint from {checkpoint_path}...")
params_whole = torch.load(checkpoint_path, map_location='cpu')
params = params_whole['net']

for key in model:
    if key in params:
        print('%s loaded' % key)
        try:
            model[key].load_state_dict(params[key])
        except:
            from collections import OrderedDict
            state_dict = params[key]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k  # remove `module.` if present
                new_state_dict[name] = v
            model[key].load_state_dict(new_state_dict, strict=False)

_ = [model[key].eval() for key in model]

# ----------------------------
# SETUP SAMPLER
# ----------------------------
sampler = DiffusionSampler(
    model.diffusion.diffusion,
    sampler=ADPM2Sampler(),
    sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
    clamp=False
)

# ----------------------------
# INFERENCE FUNCTION
# ----------------------------
def inference(text, noise, diffusion_steps=5, embedding_scale=1):
    text = text.strip()
    text = text.replace('"', '')
    ps = global_phonemizer.phonemize([text])
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)

    tokens = textclenaer(ps)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)
    
    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(tokens.device)
        text_mask = length_to_mask(input_lengths).to(tokens.device)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2) 

        s_pred = sampler(noise, 
              embedding=bert_dur[0].unsqueeze(0), num_steps=diffusion_steps,
              embedding_scale=embedding_scale).squeeze(0)

        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)

        pred_dur[-1] += 5

        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        # encode prosody
        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
        out = model.decoder((t_en @ pred_aln_trg.unsqueeze(0).to(device)), 
                                F0_pred, N_pred, ref.squeeze().unsqueeze(0))
        
    return out.squeeze().cpu().numpy()

# ----------------------------
# GENERATE AUDIO
# ----------------------------
print("Generating audio...")
start = time.time()
noise = torch.randn(1, 1, 256).to(device)
wav = inference(text, noise, diffusion_steps=5, embedding_scale=1)
rtf = (time.time() - start) / (len(wav) / 24000)
print(f"RTF = {rtf:.5f}")

# ----------------------------
# SAVE WAV
# ----------------------------
sf.write(output_wav, wav, samplerate=24000)
print(f"Saved output to {output_wav}")
