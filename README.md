# An Anchor-Free Object Detector with Contextual Information for License Plate Detection

## What does the model do?
### License plate detection & Car pose estimation!
![common_result](https://github.com/hankerkuo/Thesis_KSH/blob/master/chapters/pics/common_result.jpg)

## How's the pose estimation ability?
### Assign each pixel to certain class, possible for pixel-wise segmentation!
### (a) -> input image  (b) -> High-probability license plate position 
### (c) -> front pixels (d) -> rear pixels
![heatmap2](https://github.com/hankerkuo/Thesis_KSH/blob/master/chapters/pics/heatmap2.jpg)
![heatmap](https://github.com/hankerkuo/Thesis_KSH/blob/master/chapters/pics/heatmap.jpg)

## What kind of scenario is the model good at?
## -> A single image with multiple cars inside, see the example

### Left  column -> Vehicle-detection-based method 
### Right column -> Our method
![car_overlapped_problem](https://github.com/hankerkuo/Thesis_KSH/blob/master/chapters/pics/car_overlapped_problem.jpg)
