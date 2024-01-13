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

  The proposed framework operates in five phases to predict the class of a 3D object: A) It generates _m_ multi-view images from the 3D object. B) Feature maps are extracted from each view. C) These feature maps are converted into feature vectors, and D) importance scores are assigned based on their cosine similarity. The feature vector with the highest importance score, known as the Most Similar View (MSV), is selected as the global descriptor. E) Finally, the global descriptor is utilized to classify the object using a pre-trained classifier.

## Architecture:
![The architecture of the proposed selective multi-view deep model](/images/architecture.png "The architecture of the proposed selective multi-view deep model")

The architecture of the proposed selective multi-view deep model contains five phases:
 <br /> (A) **Multi-view extraction:** from a given 3D object, m multiple views are extracted from different viewpoints and angles. 
 <br /> (B) **Feature extraction:** each extracted view is fed to a pre-trained CNN to extract the corresponding feature stack of the detected visual features. 
 <br /> (C) **Vectorization:** the detected _m_ feature stacks are converted to _m_ feature vectors. 
 <br /> (D) **View selection:** The feature vectors are compared based on their similarity using Cosine Similarity and give a vital score that is normalized later. The more discriminative view is selected as a global descriptor based on them. 
 <br /> (D) **Object classification:** the global descriptor of the object feeds to a classifier to predict its class.

## Dataset:
[ModelNet](https://modelnet.cs.princeton.edu/) is a large-scale 3D dataset provided in 2014 by Wu et al. from Princeton University’s Computer Science Department. [ModelNet40](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Wu_3D_ShapeNets_A_2015_CVPR_paper.html) contains manually cleaned 3D objects without color information that belong to 40 class categories. In all of our experiments, and for a fair comparison, we have experimented with two versions of that dataset based on the camera settings from the literature:

* **ModelNet40v1** (Balanced and aligned dataset): in this version, the same training and testing splits of ModelNet40 as in [3dshapenets](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Wu_3D_ShapeNets_A_2015_CVPR_paper.html), [MVCNN](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/Su_Multi-View_Convolutional_Neural_ICCV_2015_paper.html), [RotationNet](https://openaccess.thecvf.com/content_cvpr_2018/html/Kanezaki_RotationNet_Joint_Object_CVPR_2018_paper.html), [DeepCCFV](https://ojs.aaai.org/index.php/AAAI/article/view/4868) were experimented. Where for each category, they used the first 80 training objects (or all if there are less than 80) for training, and balanced testing, they used the first 20 testing objects. They used the circular configuration for each object to extract the 12 aligned views. So, they ended up with 3,983 objects in 40 categories consisting of 3,183 training objects (38,196 views) and 800 testing objects (9,600 views).
  ![Circular configuration (12 views)](/images/circCameraConfig.png "Circular configuration (12 views)")
  * **ModelNet40v1 Training** can be download from [here.](https://drive.google.com/file/d/1ZTG6DkXhR0ee8tJAUkbPGncGL98t8LqS/view?usp=sharing)
  * **ModelNet40v1 Testing** can be download from [here.](https://drive.google.com/file/d/1yrNSe9YghIXm9s0kJTuzJC5oZYhrVMOe/view?usp=sharing)

  
* **ModelNet40v2** (Imbalanced and unaligned dataset): here, the whole ModelNet40 as in [RotationNet](https://openaccess.thecvf.com/content_cvpr_2018/html/Kanezaki_RotationNet_Joint_Object_CVPR_2018_paper.html), [view-GCN](https://openaccess.thecvf.com/content_CVPR_2020/html/Wei_View-GCN_View-Based_Graph_Convolutional_Network_for_3D_Shape_Analysis_CVPR_2020_paper.html), [MVTN](https://openaccess.thecvf.com/content/ICCV2021/html/Hamdi_MVTN_Multi-View_Transformation_Network_for_3D_Shape_Recognition_ICCV_2021_paper.html) were experimented. This original version of the dataset is not balanced where there is a diverse number of objects across diverse categories. It contains 12,311 3D objects split into 9,843 for training and 2,468 for testing. The literature used a spherical configuration to extract the 20 unaligned views from each object to end up with a total of 196,860 for training and 49,360 for testing.
  ![Spherical configuration (20 views)](/images/sphCameraConfig.png "Spherical configuration (20 views)")
  * **ModelNet40v2 Training** can be download from [here.](https://drive.google.com/file/d/1UiENdsOgCczr_x8-ILCA7GOpp-C1_Pqf/view?usp=sharing)
  * **ModelNet40v2 Testing** can be download from [here.](https://drive.google.com/file/d/1Vn4D3xV20fwN9SechsnIsgyfDDeZ08dv/view?usp=sharing)


Additionally, we investigate the effect of shape representation on the classification of a single view for rendering 3D objects. We utilized the ModelNet40v2 dataset for this experiment, with 12 views per 3D object. However, each 3D object was rendered using the [Phong shading technique](https://dl.acm.org/doi/pdf/10.1145/280811.280980). Shading techniques have been demonstrated to improve performance in models such as [MVDAN](https://link.springer.com/article/10.1007/s00521-021-06588-1) and [MVCNN](https://openaccess.thecvf.com/content_eccv_2018_workshops/w18/html/Su_A_Deeper_Look_at_3D_Shape_Classifiers_ECCVW_2018_paper.html). The rendered views were grayscale images with dimensions of 224 × 224 pixels and black backgrounds. The camera's field of view was adjusted so that the image canvas tightly encapsulated the 3D object.
  ![Shaded multi-view images (12 views)](/images/ShadedDataset.png "Shaded multi-view images (12 views)")
  * **ShadedModelNet40v2 Training** can be download from [here.]()
  * **ShadedModelNet40v2 Testing** can be download from [here.]()


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

