#!/usr/bin/env python
import cv2
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd
import pytesseract
import textwrap


def process_data(data, conf='72'):
    df = pd.DataFrame.from_dict(data)[['block_num', 'line_num', 'conf', 'text']]
    df = df[(df['conf'].astype(int) > int(conf)) & (df['text'].str.strip() != '')].reset_index()

    text = ''
    # -- append each line block of words
    for block, word in df.groupby(['block_num', 'line_num']):
        text += ' '.join(word['text']) + '\n'
    return text

def process_image(img, psm='3', language='eng+fra'):
    config = ('--tessdata-dir "tessdata_best" --oem 1 --dpi 72 --psm {} -l {}'.format(psm, language))
    data = pytesseract.image_to_data(img, config=config, output_type='dict')
    return data

def plot_img(ax, color_img, title):
    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]

    ax.imshow(img_RGB)
    ax.set_title(title)
    ax.set_axis_off()

def plot_text(ax, text):
    ax.text(0, 0.5, wrap(text), fontsize='small', linespacing=1.5, verticalalignment='center')
    ax.set_axis_off()

def wrap(text):
    wrapArgs = {'width': 40, 'break_long_words': True, 'replace_whitespace': False}
    fold = lambda line, wrapArgs: textwrap.fill(line, **wrapArgs)
    text = '\n'.join([fold(line, wrapArgs) for line in text.splitlines()])
    return text


def thresholding_example():
    # Create the dimensions of the figure and set title:
    fig, ((ax0,ax00,ax1,ax11,ax2,ax22),
          (ax3,ax33,ax4,ax44,ax5,ax55),
          (ax6,ax66,ax7,ax77,ax8,ax88)) = plt.subplots(nrows=3, ncols=6, sharex=True, figsize=(9, 9))
    fig.suptitle("Thresholding example", fontsize=14, fontweight='bold')
    fig.patch.set_facecolor('silver')

    # Load the image and convert it to grayscale:
    filename = 'images/anje-falkenrath.png'
    image = cv2.imread(cv2.samples.findFile(filename))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Plot the grayscale image:
    plot_img(ax0, cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "img")
    plot_text(ax00, process_data(process_image(gray_image)))

    # Apply cv2.threshold() with different thresholding values:
    ret1, thresh1 = cv2.threshold(gray_image, 60, 255, cv2.THRESH_BINARY)
    ret2, thresh2 = cv2.threshold(gray_image, 70, 255, cv2.THRESH_BINARY)
    ret3, thresh3 = cv2.threshold(gray_image, 80, 255, cv2.THRESH_BINARY)
    ret4, thresh4 = cv2.threshold(gray_image, 90, 255, cv2.THRESH_BINARY)
    ret5, thresh5 = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)
    ret6, thresh6 = cv2.threshold(gray_image, 110, 255, cv2.THRESH_BINARY)
    ret7, thresh7 = cv2.threshold(gray_image, 120, 255, cv2.THRESH_BINARY)
    ret8, thresh8 = cv2.threshold(gray_image, 130, 255, cv2.THRESH_BINARY)

    # Plot all the thresholded images:
    plot_img(ax1, cv2.cvtColor(thresh1, cv2.COLOR_GRAY2BGR), "threshold = 60")
    plot_text(ax11, process_data(process_image(thresh1)))
    plot_img(ax2, cv2.cvtColor(thresh2, cv2.COLOR_GRAY2BGR), "threshold = 70")
    plot_text(ax22, process_data(process_image(thresh2)))
    plot_img(ax3, cv2.cvtColor(thresh3, cv2.COLOR_GRAY2BGR), "threshold = 80")
    plot_text(ax33, process_data(process_image(thresh3)))
    plot_img(ax4, cv2.cvtColor(thresh4, cv2.COLOR_GRAY2BGR), "threshold = 90")
    plot_text(ax44, process_data(process_image(thresh4)))
    plot_img(ax5, cv2.cvtColor(thresh5, cv2.COLOR_GRAY2BGR), "threshold = 100")
    plot_text(ax55, process_data(process_image(thresh5)))
    plot_img(ax6, cv2.cvtColor(thresh6, cv2.COLOR_GRAY2BGR), "threshold = 110")
    plot_text(ax66, process_data(process_image(thresh6)))
    plot_img(ax7, cv2.cvtColor(thresh7, cv2.COLOR_GRAY2BGR), "threshold = 120")
    plot_text(ax77, process_data(process_image(thresh7)))
    plot_img(ax8, cv2.cvtColor(thresh8, cv2.COLOR_GRAY2BGR), "threshold = 130")
    plot_text(ax88, process_data(process_image(thresh8)))

    # Show the Figure:
    plt.show()


def thresholding_adaptive():
    # Load the image and convert it to grayscale:
    filename = 'images/anje-falkenrath.jpg'
    image = cv2.imread(cv2.samples.findFile(filename))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create the dimensions of the figure and set title:
    fig, ((axG,axGG,ax1,ax11,ax2,ax22), (axO,axOO,ax3,ax33,ax4,ax44)) = plt.subplots(nrows=2, ncols=6, sharex=True, figsize=(18, 7))
    fig.suptitle("Adaptive thresholding", fontsize=14, fontweight='bold')
    fig.patch.set_facecolor('silver')

    # Perform adaptive thresholding with different parameters:
    thresh1 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh2 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 3)
    thresh3 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh4 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 3)

    # Plot the thresholded images:
    plot_img(axG, cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "gray img")
    plot_text(axGG, process_data(process_image(gray_image)))

    plot_img(ax1, cv2.cvtColor(thresh1, cv2.COLOR_GRAY2BGR), "method=THRESH_MEAN_C, blockSize=11, C=2")
    plot_text(ax11, process_data(process_image(thresh1)))
    plot_img(ax2, cv2.cvtColor(thresh2, cv2.COLOR_GRAY2BGR), "method=THRESH_MEAN_C, blockSize=31, C=3")
    plot_text(ax22, process_data(process_image(thresh2)))

    plot_img(axO, image, "gray img")
    plot_text(axOO, process_data(process_image(image)))

    plot_img(ax3, cv2.cvtColor(thresh3, cv2.COLOR_GRAY2BGR), "method=GAUSSIAN_C, blockSize=11, C=2")
    plot_text(ax33, process_data(process_image(thresh3)))
    plot_img(ax4, cv2.cvtColor(thresh4, cv2.COLOR_GRAY2BGR), "method=GAUSSIAN_C, blockSize=31, C=3")
    plot_text(ax44, process_data(process_image(thresh4)))

    # Show the Figure:
    plt.show()



thresholding_example()
thresholding_adaptive()
