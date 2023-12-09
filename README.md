# Video-to-Video-Style-Transfer

- We aim to build upon Neural Style Transfer (NST) by extending its capabilities from static images to dynamic video content.
  
## Project Summary

- Our project extends Neural Style Transfer (NST) from static images to dynamic video content. NST merges content from one image with the style of another, creating visually striking visuals.
- Our innovation involves adapting NST for video data, extracting styles from one video and applying them to another. This opens new possibilities for artistic expression in video content, creating dynamic, style-transformed videos.
- Our approach involves developing custom neural network architectures to process video frames, extracting style features from one video and applying them seamlessly to another. The goal is to maintain artistic quality across frames, ensuring smooth style transfer in the target video.

![Flow Diagram](misc/flow.png)

## Repository Structure

1. **input_files:** Contains images and videos used as input to the models.
2. **style_images:** Holds images and videos used as styles for the model.
3. **Neural Style Transfer Model:** Python implementation of the Gatys et al neural style transfer paper.
4. **<TODO>**

## How to Run the Code

1. **Neural Style Transfer Model:**
    - Open the Model.ipynb file.
    - Specify input image/video path and style image path.
    - Run all cells of the notebook.
    - To save the stylized image, execute the `save_image` function.
    - To save the stylized video, run the last cell of the notebook.

2. **Arbitrary Image Stylization Model:**
    - <TODO>

## Outputs:

### Neural Style Transfer Model:

#### Image to Image
![Outputs](./misc/image.png)

#### Image to Video:

Input Video: 
<div>
    <video width="320" height="240" controls>
        <source src="./input_files/adwait_video.mp4" type="video/mp4">
    </video>
    <p>Input Video</p>
</div>
Style Images:
<div style="display: flex; flex-direction: row; justify-content: space-around;">
    <div>
        <img src="./style_images/style3.jpg" alt="Style1" width="320">
        <p>Style1</p>
    </div>
    <div>
        <img src="./style_images/style6.jpg" alt="Style2" width="320">
        <p>Style2</p>
    </div>
</div>
Output Videos:
<div style="display: flex; flex-direction: row; justify-content: space-around;">
<div>
        <video width="320" height="240" controls>
            <source src="./Neural Style Transfer Model/Outputs/output_adwait_style3.mp4" type="video/mp4">
        </video>
        <p>Output Video 1</p>
    </div>
    <div>
        <video width="320" height="240" controls>
            <source src="./Neural Style Transfer Model/Outputs/output_adwait.mp4" type="video/mp4">
        </video>
        <p>Output Video 2</p>
    </div>
</div>

### Arbitrary Image Stylization Model:

<TODO>
