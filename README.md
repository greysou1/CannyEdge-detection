# Canny Edge Detection

This repository is a submittion towards Programming Assignment 1 of CAP5415 - 22Fall 0V61

The detailed implementation of Canny Edge Detection in [canny_edge notebook](canny_edge.ipynb)
This notebook is also available in .py format at [canny_edge.py](canny_edge.py)

### Notebook 
The notebook goes through each step of Canny Edge Detection in each cell and the results are summarised in the notebook. 

### Python file
The python file takes the following as command line arguments

1) --img_path (str): the path where the image is stored
2) --sigma (float): the intensity of the gaussian blur
3) --LTR (float): Low Threshold Ratio for Double Threshold Hysterisis
4) --HTR (float): High Threshold Ratio for Double Threshold Hysterisis

```bash
python canny_edge.py --img_path data/67079.jpg --sigma 2.5 --LTR 0.1 --HTR 0.2
```
## Results
### Parameters
Sigma = 1.4
LTH = 0.15
HTR = 0.2
### Original image 

![org](results/original_img.jpg)
<table>
 <tr>
  <td align="center"><img src="results/gauss_x.jpg" width="100%" height="auto" /></td>
  <td align="center"><img src="results/gauss_y.jpg" width="100%" height="auto" /></td>
  <td align="center"><img src="results/X_deriv.jpg" width="100%" height="auto" /></td>
 </tr>
 <tr>
  <td align="center"><img src="results/Y_deriv.jpg" width="100%" height="auto" /></td>
  <td align="center"><img src="results/I_mag.jpg" width="100%" height="auto" /></td>
  <td align="center"><img src="results/NMS.jpg" width="100%" height="auto" /></td>
 </tr>
</table>
### Gaussian filter
![Gaussian filter](results/gaussian_filter.jpg)
### Original image with gaussian filter
![Original image with gaussian filter](results/image_with_gausssian_blur.jpg)
### gaussian derivative mask in x and y direction
![gaussian derivative ](results/derivatives_in_xy_direction.jpg)
### Magnitude and Orientation of the image
![Magnitude and Orientation](results/mag_and_orient.jpg)

### Non-Max Supression
![NMS](results/NMS.jpg)
### Double Threshold Hysterisis
![DTH](results/DTH.jpg)
### Image 1
![image 1](results/image_1.jpg)
The best sigma value is **sigma=1.5** because it eliminates suitable amount of some noise while retaining almost all important edges unlike **sigma=10**
### Image 2s
![image 2](results/image_2.jpg)
**sigma=1** is the best parameter here, as sigma is increased, the details of the person in the foreground are decreasing as well
### Image 3
![image 3](results/image_3.jpg)
Here the best **sigma=1.5** retains the edges of the building int he background while keeping the details of the boat in the foreground intact
### Image 4
![image 4](results/image_4.jpg)
In this image **sigma=1.5** is the best since it retains sharp edges and most details