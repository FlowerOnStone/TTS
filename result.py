# model vars
from TTS.utils.synthesizer import Synthesizer
from pathlib import Path
import os
import shutil
MODEL_PATH = 'tts_train_dir/vits_vivos_-April-20-2023_04+57PM-0000000/best_model.pth'
CONFIG_PATH = 'tts_train_dir/vits_vivos_-April-20-2023_04+57PM-0000000/config.json'
Speaker = "VLSP2020"


synthesizer = Synthesizer(MODEL_PATH, CONFIG_PATH)

with open("result3/testdata.txt", "r") as f:
    lines = f.readlines()
    lines = [line.split("|")[1] for line in lines]

counter = 0
for line in lines:
    counter += 1
    wav = synthesizer.tts(line, Speaker)
    synthesizer.save_wav(wav, "result3/audio{}.wav".format(counter))
