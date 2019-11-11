# An Anchor-Free Object Detector with Contextual Information for License Plate Detection

## What does the model do?
### License plate detection & Car pose estimation!
![common_result](https://github.com/hankerkuo/Thesis_KSH/blob/master/chapters/pics/common_result.jpg)

## How's the pose estimation ability?
### Assign each pixel to certain class, possible for pixel-wise segmentation!

<table>
  
<tr><td colspan="2"><strong>Results</strong></td></tr>

<!-- Line 1: (a) and (b)-->
<tr>
<td>(a) -> input image</td>
<td>(b) -> High-probability license plate position</td>
</tr>

<!-- Line 2: (c) and (d)-->
<tr>
<td>(c) -> front pixels</td>
<td>(d) -> rear pixels</td>
</tr>

</table>

![heatmap2](https://github.com/hankerkuo/Thesis_KSH/blob/master/chapters/pics/heatmap2.jpg)
![heatmap](https://github.com/hankerkuo/Thesis_KSH/blob/master/chapters/pics/heatmap.jpg)

## What kind of scenario is the model good at?
## -> An image with multiple cars inside, see the comparison below

<table>

<!-- Line 1: (a) and (b)-->
<tr>
<td><strong>Left  column -> Vehicle-detection-based method (Detection missing)</strong></td>
<td><strong>Right column -> Our method (All license plates found)</strong></td>
</tr>

</table>

![car_overlapped_problem](https://github.com/hankerkuo/Thesis_KSH/blob/master/chapters/pics/car_overlapped_problem.jpg)

## Model architecture

![modelindetail](https://github.com/hankerkuo/Thesis_KSH/blob/master/chapters/pics/modelindetail.jpg)
![Head_detail](https://github.com/hankerkuo/Thesis_KSH/blob/master/chapters/pics/Head_detail.jpg)

