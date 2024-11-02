# Automated black-and-white image colorization

Given a grayscale photograph, this project tackles the issue of fantasizing about a conceivable shading rendition of the photograph. This issue is underconstrained, so past methodologies have either depended on significant human cooperation or came about in desaturated colorizations. We propose a wholly automated approach that produces lively furthermore, sensible colorizations. This approach utilizes the combination of CNN-based colorization and parameter-free k-means clustering to identify the color spills, so that the image can be recolored to produce color images that are aesthetically more pleasing and plausible than the images produced by the state-of-the-art methods. The performance of the proposed model is evaluated in terms of mean square error and structural similarity and found to be superior to the related works.

Terminology:

- Model-1: Proposed by Zhang et al. in their paper Colorful Image Colorization
- Model-2: Proposed by Zhang et al. in their paper User-Guided Image Colorization

The results produced by state-of-the-art automated image colorization (Model-1) still lacked spatial consistency for a large number of images and the results by Model-2 are comparatively more consistent but the approach is not fully automated. So our main aim was to achieve results that are
spatially more consistent and are fully automated at the same time. Therefore, instead of training a new model from scratch, we went with the approach in which we combined the above two works to achieve our objective.

### Steps:

1. We extract the l channel of the image and give it as input to Model-1. The model gives a and b channel as output which are then combined with the input l channel to produce the colored image.

2) A set of patches(3x3) with their respective a and b channels as extracted from the colored image generated in Step-1 are then given as input to Model-2 along with the black and white image. The model then generates the image which is more spatially consistent then the one generated in Step-1.
   Model-2 produces a default colorized output even if no user input is given i.e. no color patches are given as input. We used the output of Step-1 to choose color patches because the default output of Model-2 although spatially consistent, tends to be desaturated. Thus, the color patches were taken to ensure that the colorization results are bright.

Check [_Scripts_](./Scripts/) folder for implementation.

Published Springer Article: https://link.springer.com/article/10.1007/s11760-021-02047-5

[**Results**](./output.pdf)

<img width="766" alt="image" src="https://github.com/user-attachments/assets/6217102e-22ab-4724-b7da-9cbc76965339">
