# Hyperspectral-Image-Classification-Based-on-Gramian-Angular-Fields-Encoding

Abstract：
Using convolutional neural networks (CNN) as a classifier has proven effective in hyperspectral image classifica- tion. However, overfitting is a common problem that CNN may face when spectral and spatial information are integrated into the input patch for modeling. Unlike previous approaches, a new feature processing method based on Gramian Angular Fields encoding is proposed in this paper. This specifically focuses on improving edge pixel accuracy for different classes by encoding the 1-D spectral feature into the 2-D Gramian matrix as the data samples that are independent to each other. Without the disturbance from neighboring noises, the spectral information can enhance the classification performance for each pixel using the proposed transformation scheme. Experiments show that the proposed method achieve higher accuracy on edge pixels than other CNN classification approaches.

The transformation process from 1-D series HSI data to the 2D Gramian matrix image as shown below,

![WechatIMG2179](https://user-images.githubusercontent.com/60961564/203110547-3ec47793-82bb-4719-88ca-57b5669190db.png)

(a) HSI reflectance spectrum for one pixel. (b) Illustration of the encoding series from the Cartesian coordinate system to the polar coordinate system.

After the reflectance spectrum series θ is represented in polar form, the 1D to 2D transformation for HSI pixel vector θ is accomplished by generating the following matrix,


![WechatIMG2182](https://user-images.githubusercontent.com/60961564/203117019-dc9f2b5e-3c55-4ff0-98f8-f8eb71331b17.png)


This is the Gramian matrix to describe the desired 2D feature image as shown below,


![WechatIMG2183](https://user-images.githubusercontent.com/60961564/203120285-593955a8-a312-4bf6-a036-4f53138af092.png)

We apply efficientNet as a basic classifier to classify each class with the 2D feature images of HSI as the data samples. The pixel classification result as shown below on the IP dataset,


![WechatIMG2184](https://user-images.githubusercontent.com/60961564/203121580-31a79bd5-43aa-4cfb-9eb8-0f4124165e1d.png)
