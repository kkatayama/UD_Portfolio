# coding: utf-8
from glob import glob

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import pytesseract


def process_data(data, conf, prob=False):
    df = pd.DataFrame.from_dict(data)[['block_num', 'line_num', 'conf', 'text']]
    df["conf"] = df["conf"].apply(lambda x: int(float(x)))
    conf = int(conf)
    df = df[(df['conf'] > conf) & (df['text'].str.strip() != '')].reset_index()

    # -- print text
    print('--- OCR TEXT ---')
    for block, word in df.groupby(['block_num', 'line_num']):
        print(' '.join(word['text']))

    # -- print text (single line)
    print('\n--- OCR TEXT (single line)---')
    print(' '.join(df.text))

    if prob:
        # print probabilitites
        print('\n--- OCR CONFIDENCE SCORES ---')
        for block_no, (block, word) in enumerate(df.groupby(['block_num', 'line_num'])):
            print('block #{} | block: {}'.format(block_no, block))
            print_df(word[['text', 'conf']].transpose())

def process_image(img, process_type, psm, language):
    config = ('--tessdata-dir "tessdata_best" --oem 1 --dpi 72 --psm {} -l {}'.format(psm, language))
    text = pytesseract.image_to_string(img, config=config, output_type='dict')
    data = pytesseract.image_to_data(img, config=config, output_type='dict')
    return text, data

def crop(im, H, W, TB, LR):
    img = im[name_TB:(name_TB+name_H), name_LR:(name_W+name_LR)]
    cv2.imshow('image_name', img)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()
    return img


if __name__ == '__main__':
    for filename in glob('images/*'):
        print('\n=== {} ===\n'.format(filename.split('/')[-1]))
        image = cv2.imread(cv2.samples.findFile(filename))
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # -- image size (whole card)
        height, width = gray_image.shape

        # -- name block
        name_H  = int(0.0585 * height)
        name_W  = int(0.6950 * width)
        name_TB = int(0.0455 * height)
        name_LR = int(0.0755 * width)

        # -- crop image
        image_name = crop(gray_image, name_H, name_W, name_TB, name_LR)

        # -- ocr image
        text, data = process_image(image_name, '', '7', 'eng+fra')
        process_data(data, '72')


