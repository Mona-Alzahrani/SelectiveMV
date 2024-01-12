# SelectiveMV: Selective Multi-View Deep Model for 3D Object Classification (SelectiveMV)
This repository is for the following paper _"Selective Multi-View Deep Model for 3D Object Classification (SelectiveMV)"_ introduced by [Mona Alzahrani](https://github.com/Mona-Alzahrani), Muhammad Usman, [Saeed Anwar](https://github.com/saeed-anwar), and Tarek Helmy, published in I3D 2024.

## Requirements: 
The model is built in _Visual Studio Code_ editor using: 
* Tensorflow-gpu 2.10
* Cuda 11.2
* cuDNN8.1 
* Python 3.9 

## Content:
1. [Introduction](#introduction)
2. [Architecture](#architecture)
3. [Dataset](#dataset)
4. [Training](#training)
5. [Testing](#testing)
6. [Results](#results)

## Introduction:
3D object classification has emerged as a practical technology with applications in various domains, such as medical image analysis, automated driving, intelligent robots, and crowd surveillance. Among the different approaches, multi-view representations for 3D object classification has shown the most promising results, achieving state-of-the-art performance. However, there are certain limitations in current view-based 3D object classification methods. One observation is that using all captured views for classifying 3D objects can confuse the classifier and lead to misleading results for certain classes. Additionally, some views may contain more discriminative information for object classification than others. These observations motivate the development of smarter and more efficient selective multi-view classification models. In this work, we propose a Selective Multi-View Deep Model that extracts multi-view images from 3D data representations and selects the most influential view by assigning importance scores using the cosine similarity method based on visual features detected by a pre-trained CNN. The proposed method is evaluated on the ModelNet40 dataset for the task of 3D classification. The results demonstrate that the proposed model achieves an overall accuracy of 88.13% using only a single view when employing a shading technique for rendering the views, pre-trained ResNet-152 as the backbone CNN for feature extraction, and a Fully Connected Network (FCN) as the classifier.

![Illustration of the proposed framework](/images/framework.png "Illustration of the proposed framework")

  The proposed framework operates in five phases to predict the class of a 3D object: A) It generates N multi-view images from the 3D object. B) Feature maps are extracted from each view. C) These feature maps are converted into feature vectors, and D) importance scores are assigned based on their cosine similarity. The feature vector with the highest importance score, known as the Most Similar View (MSV), is selected as the global descriptor. E) Finally, the global descriptor is utilized to classify the object using a pre-trained classifier.

## Architecture:
## Dataset:
## Training:
## Testing:
## Results:


## Citation:
If you find the code helpful in your resarch or work, please cite the following paper:
```
@article{SelectiveMV2024,
  title={Selective Multi-View Deep Model for 3D Object Classification},
  author={Alzahrani, Mona and Usman, Muhammad and Anwar, Saeed and Helmy, Tarek},
  journal={Proceedings of the ACM on Computer Graphics and Interactive Techniques},
  volume={2},
  number={2},
  pages={1--16},
  year={2024},
  publisher={ACM New York, NY, USA}
}
```

## Acknowledgement:
This project is funded by the Interdisciplinary Research Center for Intelligent Secure Systems at King Fahd University of Petroleum & Minerals (KFUPM) under Grant Number INSS2305.

