# coding: utf-8
# %load thresholding_tests.py
import cv2
import matplotlib
import matplotlib.pyplot as plt
# from skimage.data import page
from skimage.filters import (threshold_otsu, threshold_triangle, threshold_niblack, threshold_sauvola)
from skimage import img_as_ubyte


def show_img_with_matplotlib(color_img, title, pos, num_rows=2):
    """Shows an image using matplotlib capabilities"""

    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(num_rows, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


def show_hist_with_matplotlib_gray(hist, title, pos, color, t=-1, num_rows=2):
    """Shows the histogram using matplotlib capabilities"""

    ax = plt.subplot(num_rows, 2, pos)
    # plt.title(title)
    plt.xlabel("bins")
    plt.ylabel("number of pixels")
    plt.xlim([0, 256])
    plt.axvline(x=t, color='m', linestyle='--')
    plt.plot(hist, color=color)


def thresholding_adaptive():
    # Load the image and convert it to grayscale:
    image = cv2.imread(cv2.samples.findFile(args['image']))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create the dimensions of the figure and set title:
    fig = plt.figure(figsize=(15, 7))
    plt.suptitle("Adaptive thresholding", fontsize=14, fontweight='bold')
    fig.patch.set_facecolor('silver')

    # Perform adaptive thresholding with different parameters:
    thresh1 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh2 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 3)
    thresh3 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh4 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 3)

    # Plot the thresholded images:
    show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "gray img", 1)
    show_img_with_matplotlib(cv2.cvtColor(thresh1, cv2.COLOR_GRAY2BGR), "method=THRESH_MEAN_C, blockSize=11, C=2", 2)
    show_img_with_matplotlib(cv2.cvtColor(thresh2, cv2.COLOR_GRAY2BGR), "method=THRESH_MEAN_C, blockSize=31, C=3", 3)
    show_img_with_matplotlib(cv2.cvtColor(thresh3, cv2.COLOR_GRAY2BGR), "method=GAUSSIAN_C, blockSize=11, C=2", 5)
    show_img_with_matplotlib(cv2.cvtColor(thresh4, cv2.COLOR_GRAY2BGR), "method=GAUSSIAN_C, blockSize=31, C=3", 6)

    # Show the Figure:
    plt.show()


def thresholding_adaptive_filter_noise():
    # Create the dimensions of the figure and set title and color:
    fig = plt.figure(figsize=(15, 7))
    plt.suptitle("Adaptive thresholding applying a bilateral filter (noise removal while edges sharp)", fontsize=14,
                 fontweight='bold')
    fig.patch.set_facecolor('silver')

    # Load the image and convert it to grayscale:
    image = cv2.imread(cv2.samples.findFile(args['image']))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a bilateral filter in order to reduce noise while keeping the edges sharp:
    gray_image = cv2.bilateralFilter(gray_image, 15, 25, 25)

    # Perform adaptive thresholding with different parameters:
    thresh1 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh2 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 3)
    thresh3 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh4 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 3)

    # Plot the thresholded images:
    show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "gray img", 1)
    show_img_with_matplotlib(cv2.cvtColor(thresh1, cv2.COLOR_GRAY2BGR), "method=THRESH_MEAN_C, blockSize=11, C=2", 2)
    show_img_with_matplotlib(cv2.cvtColor(thresh2, cv2.COLOR_GRAY2BGR), "method=THRESH_MEAN_C, blockSize=31, C=3", 3)
    show_img_with_matplotlib(cv2.cvtColor(thresh3, cv2.COLOR_GRAY2BGR), "method=GAUSSIAN_C, blockSize=11, C=2", 5)
    show_img_with_matplotlib(cv2.cvtColor(thresh4, cv2.COLOR_GRAY2BGR), "method=GAUSSIAN_C, blockSize=31, C=3", 6)

    # Show the Figure:
    plt.show()


def thresholding_example():
    # Create the dimensions of the figure and set title and color:
    fig = plt.figure(figsize=(9, 9))
    plt.suptitle("Thresholding example", fontsize=14, fontweight='bold')
    fig.patch.set_facecolor('silver')

    # Load the image and convert it to grayscale:
    image = cv2.imread(cv2.samples.findFile(args['image']))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Plot the grayscale image:
    show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "img", 1)

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
    show_img_with_matplotlib(cv2.cvtColor(thresh1, cv2.COLOR_GRAY2BGR), "threshold = 60", 2, num_rows=3)
    show_img_with_matplotlib(cv2.cvtColor(thresh2, cv2.COLOR_GRAY2BGR), "threshold = 70", 3, num_rows=3)
    show_img_with_matplotlib(cv2.cvtColor(thresh3, cv2.COLOR_GRAY2BGR), "threshold = 80", 4, num_rows=3)
    show_img_with_matplotlib(cv2.cvtColor(thresh4, cv2.COLOR_GRAY2BGR), "threshold = 90", 5, num_rows=3)
    show_img_with_matplotlib(cv2.cvtColor(thresh5, cv2.COLOR_GRAY2BGR), "threshold = 100", 6, num_rows=3)
    show_img_with_matplotlib(cv2.cvtColor(thresh6, cv2.COLOR_GRAY2BGR), "threshold = 110", 7, num_rows=3)
    show_img_with_matplotlib(cv2.cvtColor(thresh7, cv2.COLOR_GRAY2BGR), "threshold = 120", 8, num_rows=3)
    show_img_with_matplotlib(cv2.cvtColor(thresh8, cv2.COLOR_GRAY2BGR), "threshold = 130", 9, num_rows=3)

    # Show the Figure:
    plt.show()


def thresholding_otsu_filter_noise():
    # Create the dimensions of the figure and set title and color:
    fig = plt.figure(figsize=(11, 10))
    plt.suptitle("Otsu's binarization algorithm applying a Gaussian filter", fontsize=14, fontweight='bold')
    fig.patch.set_facecolor('silver')

    # Load the image and convert it to grayscale:
    image = cv2.imread(cv2.samples.findFile(args['image']))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the histogram
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

    # Otsu's binarization algorithm:
    ret1, th1 = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #  Blurs the image using a Gaussian filter to eliminate noise
    gray_image_blurred = cv2.GaussianBlur(gray_image, (25, 25), 0)

    # Calculate histogram after filtering:
    hist2 = cv2.calcHist([gray_image_blurred], [0], None, [256], [0, 256])

    # Otsu's binarization algorithm:
    ret2, th2 = cv2.threshold(gray_image_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Plot all the images:
    show_img_with_matplotlib(image, "image with noise", 1, num_rows=3)
    show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "gray img with noise", 2, num_rows=3)
    show_hist_with_matplotlib_gray(hist, "grayscale histogram", 3, 'm', ret1, num_rows=3)
    show_img_with_matplotlib(cv2.cvtColor(th1, cv2.COLOR_GRAY2BGR),
                             "Otsu's binarization (before applying a Gaussian filter)", 4, num_rows=3)
    show_hist_with_matplotlib_gray(hist2, "grayscale histogram", 5, 'm', ret2, num_rows=3)
    show_img_with_matplotlib(cv2.cvtColor(th2, cv2.COLOR_GRAY2BGR),
                             "Otsu's binarization (after applying a Gaussian filter)", 6, num_rows=3)

    # Show the Figure:
    plt.show()


def thresholding_otsu_ex():
    # Create the dimensions of the figure and set title and color:
    fig = plt.figure(figsize=(10, 10))
    plt.suptitle("Otsu's binarization algorithm", fontsize=14, fontweight='bold')
    fig.patch.set_facecolor('silver')

    # Load the image and convert it to grayscale:
    image = cv2.imread(cv2.samples.findFile(args['image']))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate histogram (only for visualization):
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

    # Threshold the image aplying Otsu's algorithm:
    ret1, th1 = cv2.threshold(gray_image, 0, 255, cv2.THRESH_TRUNC + cv2.THRESH_OTSU)

    # Plot all the images:
    show_img_with_matplotlib(image, "image", 1)
    show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "gray img", 2)
    show_hist_with_matplotlib_gray(hist, "grayscale histogram", 3, 'm', ret1, num_rows=2)
    show_img_with_matplotlib(cv2.cvtColor(th1, cv2.COLOR_GRAY2BGR), "Otsu's binarization", 4)

    # Show the Figure:
    plt.show()


def thresholding_scikit():
    # Create the dimensions of the figure and set title:
    fig = plt.figure(figsize=(12, 8))
    plt.suptitle("Thresholding scikit-image (Otsu, Triangle, Niblack, Sauvola)", fontsize=14, fontweight='bold')
    fig.patch.set_facecolor('silver')

    # Load the image and convert it to grayscale:
    image = cv2.imread(cv2.samples.findFile(args['image']))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate histogram (only for visualization):
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

    # Trying Otsu's scikit-image algorithm:
    thresh_otsu = threshold_otsu(gray_image)
    binary_otsu = gray_image > thresh_otsu
    binary_otsu = img_as_ubyte(binary_otsu)

    # Trying Niblack's scikit-image algorithm:
    thresh_niblack = threshold_niblack(gray_image, window_size=25, k=0.8)
    binary_niblack = gray_image > thresh_niblack
    binary_niblack = img_as_ubyte(binary_niblack)

    # Trying Sauvola's scikit-image algorithm:
    thresh_sauvola = threshold_sauvola(gray_image, window_size=25)
    binary_sauvola = gray_image > thresh_sauvola
    binary_sauvola = img_as_ubyte(binary_sauvola)

    # Trying triangle scikit-image algorithm:
    thresh_triangle = threshold_triangle(gray_image)
    binary_triangle = gray_image > thresh_triangle
    binary_triangle = img_as_ubyte(binary_triangle)

    # Plot all the images:
    show_img_with_matplotlib(image, "image", 1)
    show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "gray img", 2)
    show_img_with_matplotlib(cv2.cvtColor(binary_otsu, cv2.COLOR_GRAY2BGR), "Otsu's binarization (scikit-image)", 3)
    show_img_with_matplotlib(cv2.cvtColor(binary_triangle, cv2.COLOR_GRAY2BGR), "Triangle binarization (scikit-image)", 4)
    show_img_with_matplotlib(cv2.cvtColor(binary_niblack, cv2.COLOR_GRAY2BGR), "Niblack's binarization (scikit-image)", 5)
    show_img_with_matplotlib(cv2.cvtColor(binary_sauvola, cv2.COLOR_GRAY2BGR), "Sauvola's binarization (scikit-image)", 6)

    # Show the Figure:
    plt.show()


def thresholding_scikit2():
    matplotlib.rcParams['font.size'] = 9
    image = cv2.imread(cv2.samples.findFile(args['image']))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    binary_global = gray_image > threshold_otsu(gray_image)

    window_size = 25
    thresh_niblack = threshold_niblack(gray_image, window_size=window_size, k=0.8)
    thresh_sauvola = threshold_sauvola(gray_image, window_size=window_size)

    binary_niblack = gray_image > thresh_niblack
    binary_sauvola = gray_image > thresh_sauvola

    plt.figure(figsize=(8, 7))
    plt.subplot(2, 2, 1)
    plt.imshow(gray_image, cmap=plt.cm.gray)
    plt.title('Original')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title('Global Threshold')
    plt.imshow(binary_global, cmap=plt.cm.gray)
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(binary_niblack, cmap=plt.cm.gray)
    plt.title('Niblack Threshold')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(binary_sauvola, cmap=plt.cm.gray)
    plt.title('Sauvola Threshold')
    plt.axis('off')

    plt.show()


def thresholding_adaptive2():
    img = cv2.imread(cv2.samples.findFile(args['image']))
    grayscaled = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)

    cv2.imshow('original',grayscaled)
    cv2.imshow('Adaptive threshold',th)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

i = 3
args = {
    'image': 'images/anje-falkenrath.png',
    'preprocess': 'blur',
    'psm': '{}'.format(i),
    'lang': 'eng+fra',
    'conf': '72',
    'prob': False
}

#thresholding_scikit()
#thresholding_adaptive()
