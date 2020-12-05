# UD_Portfolio

## Scripts
#### thresholding_tests.py
> Applies different image processing techniques to an image

* show_img_with_matplotlib(color_img, title, pos, num_rows=2)
* > add image plot to figure
  * color_img - binary image to plot
  * title - title for the plot
  * pos - column position for figure
  * num_rows - number of rows in figure
 
* show_hist_with_matplotlib_gray(hist, pos, color, t=-1, num_rows=2)
  > add histogram plot to figure 
  * hist - a []256x1] array where each value corresponds to number of pixels in that image with its corresponding pixel value
  * pos - column position for figure
  * color - line color
  * t - position in data coordinates of the vertical line 
  * num_rows - number of rows in figure

* thresholding_adaptive()
  > apply [cv2.adaptiveThreshold](https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#ga72b913f352e4a1b1b397736707afcde3) with varying parameters

* thresholding_adaptive_filter_nois()
  > apply [cv2.bilateralFilter](https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#ga9d7064d478c95d60003cf839430737ed) to image before applying [cv2.adaptiveThreshold](https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#ga72b913f352e4a1b1b397736707afcde3)

* thresholding_example()
  > apply [cv2.threshold](https://docs.opencv.org/master/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57) with [cv2.THRESH_BINARY](https://docs.opencv.org/master/d7/d1b/group__imgproc__misc.html#ggaa9e58d2860d4afa658ef70a9b1115576a147222a96556ebc1d948b372bcd7ac59)

* thresholding_otsu_ex()
  > applying "Otsu's binarization algorithm" with [cv.THRESH_OTSU](https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#ggaa9e58d2860d4afa658ef70a9b1115576a95251923e8e22f368ffa86ba8bce87ff) 

* thresholding_otsu_filter_noise()_
  > apply Gaussian filter before applying Otsu's binarization algorithm

* thresholding_scikit()
  > applying some [image filters](https://scikit-image.org/docs/dev/api/skimage.filters.html) using the [scikit-image](https://scikit-image.org/) instead of opencv

* thresholding_scikit2()
  > deprecated function

* thresholding_adaptive2()
  > deprecated function

#### img_processing_test.py
> Applying various image processing techniques along side tesseract OCR

* process_data(data, conf='72')
* process_image(img, psm='3', language='eng+fra')
* plot_img(ax, color_img, title)
* plot_text(ax, text)
* 

#### mtg_ocr.py
> Crop an image prior to applying OCR
> Currently only "name" implemented

* process_data(data, conf, prob=False)
  > Parse the Tesseract OCR results and print
  * data - 
  * conf - 
  * prob - 
  
* process_image(img, process_type, psm, language)
  > Send image to Tesseract OCR for processing
  * RETURNS: (text, data)
    * text - 
    * data - 
    
* crop(im, H, W, TB, LR)
  > Crop an image with provided parameters
  * im - 
  * H - 
  * W - 
  * TB - 
  * LR - 
  
  
#### scrape_img_text.py
> Apply Tesseract OCR to an image and print the results
```bash
usage: scrape_img_text.py [-h] --image IMAGE [--preprocess PREPROCESS] [--psm PSM] [--lang LANG] [--conf CONF] [--prob]

optional arguments:
  -h, --help            show this help message and exit
  --image IMAGE         path to input image to be OCR'd
  --preprocess PREPROCESS
                        thresh: preprocess using threshold method
                        blur: preprocess using blur method
  --psm PSM             Page segmentation mode
  --lang LANG           language
  --conf CONF           confidence threshold
  --prob                show probabilitites
```
* print_df(_df)
  > 
* process_image(image_file, process_type, psm, language)
  > 
