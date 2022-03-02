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
 
* show_hist_with_matplotlib_gray(hist, title, pos, color, t=-1, num_rows=2)
  > add histogram plot to figure 
  * hist - a []256x1] array where each value corresponds to number of pixels in that image with its corresponding pixel value
  * pos - column position for figure
  * color - line color
  * t - position in data coordinates of the vertical line 
  * num_rows - number of rows in figure

* thresholding_adaptive()
  > apply [cv2.adaptiveThreshold](https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#ga72b913f352e4a1b1b397736707afcde3) with varying parameters
  ![adaptive_thresholding](https://raw.githubusercontent.com/kkatayama/UD_Portfolio/main/readme_images/adaptive_thresholding.png)
  

* thresholding_adaptive_filter_noise()
  > apply [cv2.bilateralFilter](https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#ga9d7064d478c95d60003cf839430737ed) to image before applying [cv2.adaptiveThreshold](https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#ga72b913f352e4a1b1b397736707afcde3)
  ![adaptive_thresholding_filter_noise](https://raw.githubusercontent.com/kkatayama/UD_Portfolio/main/readme_images/adaptive_thresholding_bilateral_filter.png)

* thresholding_example()
  > apply [cv2.threshold](https://docs.opencv.org/master/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57) with [cv2.THRESH_BINARY](https://docs.opencv.org/master/d7/d1b/group__imgproc__misc.html#ggaa9e58d2860d4afa658ef70a9b1115576a147222a96556ebc1d948b372bcd7ac59)
  ![thresholding_range_60_130](https://github.com/kkatayama/UD_Portfolio/blob/main/readme_images/thresholding_range_60_130.png)

* thresholding_otsu_ex()
  > applying "Otsu's binarization algorithm" with [cv.THRESH_OTSU](https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#ggaa9e58d2860d4afa658ef70a9b1115576a95251923e8e22f368ffa86ba8bce87ff) 
  ![otsu_binarization](https://github.com/kkatayama/UD_Portfolio/blob/main/readme_images/otsu_binarization.png)

* thresholding_otsu_filter_noise()_
  > apply Gaussian filter before applying Otsu's binarization algorithm
  ![otsu_gaussian](https://github.com/kkatayama/UD_Portfolio/blob/main/readme_images/otsu_gaussian.png)

* thresholding_scikit()
  > applying some [image filters](https://scikit-image.org/docs/dev/api/skimage.filters.html) using the [scikit-image](https://scikit-image.org/) instead of opencv
  ![thresholding_scikit](https://github.com/kkatayama/UD_Portfolio/blob/main/readme_images/thresholding_using_scikit.png)

* thresholding_scikit2()
  > deprecated function

* thresholding_adaptive2()
  > deprecated function
  

#### img_processing_test.py
> Applying various image processing techniques along side tesseract OCR

#### Helper Functions
* process_data(data, conf='72')
* process_image(img, psm='3', language='eng+fra')
* plot_img(ax, color_img, title)
* plot_text(ax, text)

#### Processing Functions
* thresholding_example()
  > Apply the previous thresholding example on a card image and then run OCR.
  ![applied_thresholding](https://github.com/kkatayama/UD_Portfolio/blob/main/readme_images/applied_thresholding.png)

* thresholding_adaptive()
  > Apply the previous adaptive thresholding example on a card image and then run OCR.
  ![applied_adaptive_thresholding](https://github.com/kkatayama/UD_Portfolio/blob/main/readme_images/applied_adaptive_thresholding.png)

#### mtg_ocr.py
> Crop an image prior to applying OCR
> Currently only "name" implemented

* process_data(data, conf, prob=False)
  > Parse the Tesseract OCR results and print
  * data - object generated from `pytesseract.image_to_data(img, config=config, output_type='dict')`
  * conf - confidence threshold (default = 72)
  * prob - option to print probabilitites (default = False)
  
* process_image(img, process_type, psm, language)
  > Send image to Tesseract OCR for processing
  * RETURNS: (text, data)
    * text - pytesseract.image_to_string(img, config=config, output_type='dict')
    * data - pytesseract.image_to_data(img, config=config, output_type='dict')
    
* crop(im, H, W, TB, LR)
  > Crop an image with provided parameters
  * im - opencv image object
  *  H - height
  *  W - width
  * TB - top_to_bottom
  * LR - left_to_right
  
  
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

#### Sample Run
```bash
python scrape_img_text.py --preprocess blur --psm 3 --lang eng+fra --image images/anje-falkenrath.png --conf 82
```
```asciidoc
--- OCR TEXT ---
Anje Falkenrath
Legendary Creature — Vampire
Discard a card: Draw a card.
Whenever you discard a card, if it has
madness, untap Anje Falkenrath.
“We all hide a little madness behind our
sophistication, do we not?”
037/302 M
EN CYNTHIA SHEPPARD & © 2019 Wizards of the Coast

--- OCR TEXT (single line)---
Anje Falkenrath Legendary Creature — Vampire Discard a card: Draw a card. Whenever you discard a card, if it has madness, untap Anje Falkenrath. “We all hide a little madness behind our sophistication, do we not?” 037/302 M EN CYNTHIA SHEPPARD & © 2019 Wizards of the Coast
```
![anje_falkenrath](https://github.com/kkatayama/UD_Portfolio/blob/main/images/anje-falkenrath.png)
