#!/usr/bin/env python3
'''
USAGE: python scrape_img_test.py --image [FILE] --preprocess [blur, thresh] --psm [0-13] --lang [language]

python scrape_img_test.py --image images/anje-falkenrath.jpg
python scrape_img_test.py --image images/anje-falkenrath.jpg  --preprocess blur
python scrape_img_test.py --image images/anje-falkenrath.jpg  --preprocess thresh --psm 3
python scrape_img_test.py --image images/anje-falkenrath.jpg  --preprocess thresh --psm 3 --lang eng+enm
python scrape_img_test.py --image images/anje-falkenrath.jpg  --preprocess thresh --psm 3 --lang eng+enm --conf 72
python scrape_img_test.py --image images/anje-falkenrath.jpg  --preprocess thresh --psm 3 --lang eng+enm --conf 72 --prob
'''

from tabulate import tabulate
from PIL import Image
import pytesseract
import argparse
import cv2
import os
import pandas as pd
import argparse


def print_df(_df):
    print(tabulate(_df, tablefmt='fancy_grid'))


def process_image(image_file, process_type, psm, language):
    # load the example image and convert it to grayscale
    image = cv2.imread(image_file)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # cv2.imshow("Image", gray)

    # check to see if we should apply thresholding to preprocess the
    # image
    if process_type == "thresh":
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # make a check to see if median blurring should be done to remove
    # noise
    elif process_type == "blur":
        gray = cv2.medianBlur(gray, 3)

    # write the grayscale image to disk as a temporary file so we can
    # apply OCR to it
    filename = os.getcwd() + os.path.sep + "tmp" + os.path.sep + "{}.png".format(os.getpid())
    cv2.imwrite(filename, gray)

    # load the image as a PIL/Pillow image, apply OCR, and then delete
    # the temporary file
    config = ('--tessdata-dir "tessdata_best" --oem 1 --dpi 72 --psm {} -l {}'.format(psm, language))
    text = pytesseract.image_to_string(Image.open(filename), config=config, output_type='dict')
    data = pytesseract.image_to_data(Image.open(filename), config=config, output_type='dict')
    os.remove(filename)

    # show the output images
    # cv2.imshow("Image", image)
    # cv2.imshow("Output", gray)
    # cv2.waitKey(0)
    return text, data


if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--image",
                    required=True,
                    help="path to input image to be OCR'd")
    ap.add_argument("--preprocess",
                    default="thresh",
                    help="thresh: preprocess using threshold method\nblur: preprocess using blur method")
    ap.add_argument("--psm",
                    default="3",
                    help="Page segmentation mode")
    ap.add_argument("--lang",
                    default="eng",
                    help="language")
    ap.add_argument("--conf",
                    default="70",
                    help="confidence threshold")
    ap.add_argument("--prob",
                    default=False,
                    action='store_true',
                    help="show probabilitites")
    args = vars(ap.parse_args())

    # -- text, data = process_image('images/anje-falkenrath.jpg', 'thresh', '3', 'eng')
    text, data = process_image(args["image"], args["preprocess"], args["psm"], args["lang"])

    df = pd.DataFrame.from_dict(data)[['block_num', 'line_num', 'conf', 'text']]
    df = df[(df['conf'].astype(int) > int(args["conf"])) & (df['text'].str.strip() != '')].reset_index()

    # -- print text
    print('--- OCR TEXT ---')
    for block, word in df.groupby(['block_num', 'line_num']):
        print(' '.join(word['text']))

    # -- print text (single line)
    print('\n--- OCR TEXT (single line)---')
    print(' '.join(df.text))

    if args["prob"]:
        # print probabilitites
        print('\n--- OCR CONFIDENCE SCORES ---')
        for block_no, (block, word) in enumerate(df.groupby(['block_num', 'line_num'])):
            print('block #{} | block: {}'.format(block_no, block))
            print_df(word[['text', 'conf']].transpose())
