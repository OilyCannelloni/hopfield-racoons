from PIL import Image
import os

RAW_PATH = ".\\pigs_predict_raw"
DEST_PATH = ".\\pigs_predict"

if __name__ == '__main__':
    raw_dir = os.fsencode(RAW_PATH)
    i = 1
    for file in os.listdir(raw_dir):
        filename = os.fsdecode(file)
        filename = f"{RAW_PATH}\\{filename}"
        img = Image.open(filename)
        img = img.resize((100, 100))
        img = img.convert('1')
        img.save(f"{DEST_PATH}\\pig{i}.bmp")
        i += 1