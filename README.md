# Multi-Task Learning U-Net for Functional Shoulder Sub-Task Segmentation

## Abstract
The assessment of a frozen shoulder (FS) is critical for evaluating outcomes and medical treatment. Analysis of functional shoulder sub-tasks provides more crucial information, but current manual labeling methods are time-consuming and prone to errors. To address this challenge, we propose a deep multi-task learning (MTL) U-Net to provide an automatic and reliable functional shoulder sub-task segmentation (STS) tool for clinical evaluation in FS. The proposed approach contains the main task of STS and the auxiliary task of transition point detection (TPD). For the main STS task, a U-Net architecture including an encoder-decoder with skip connection is presented to perform shoulder sub-task classification for each time point. The auxiliary TPD task uses lightweight convolutional neural networks architecture to detect the boundary between shoulder sub-tasks. A shared structure is implemented between two tasks and their objective functions of them are optimized jointly. The fine-grained transition-related information from the auxiliary TPD task is expected to help the main STS task better detect boundaries between functional shoulder sub-tasks. We conduct the experiments using wearable inertial measurement units to record 815 shoulder task sequences collected from 20 healthy subjects and 43 patients with FS. The experimental results present that the deep MTL U-Net can achieve superior performance compared to using single-task models. It shows the effectiveness of the proposed method for functional shoulder STS.

## Data preprocessing
Before fed into the network, all time-serial data is first denoised with simple moving average (SMA) filter by averaging a group of samples. A 5-point SMA filter is calculated as the equation below, where $\tilde{x}$ is the filtered sample, $x$ is the original sample, and $n$ is the sample index. The example smoothed time-serial data is presented in Fig. 1 (a).

$$\tilde{x}_n = \frac{x_{n-2}+x_{n-1}+x_n+x_{n+1}+x_{n+2}}{5}$$

Next, we apply zero-padding to each filtered time-serial sequence $\tilde{X}$ for length resizing. This resizing process ensures all sequences to the same size. Let $l_i$ be the length of sequence $i$, and $l_{max}$ is the maximum of $\\{l_i│\forall i \in [1,n]\\}$, where $n$ is the total number of sequences. Zero values are added before and after each original time-serial sequence to ensure the new sequence $\hat{X}$ have the same length equal to $l_{max}$ , as shown in Fig. 1 (b).

The added zero samples are labeled as a new sub-task class to be distinguished from the original shoulder sub-task samples. The sub-task boundaries in each IMU sequence are normalized with a respect to $l_{max}$ . The resized sequence of class labels $C$ and the set of transition points $P$ for $\hat{X}$ are illustrated in Fig. 1 (c).

<p align="center"><img src="https://user-images.githubusercontent.com/102669387/217750074-4cbdceb9-d3e7-48a0-ae34-a58807a9cc1a.png" width=40% height=40%><br/>(a)<br/><img src="https://user-images.githubusercontent.com/102669387/217750086-a319831c-6619-4105-8f8f-2121b19d6895.png" width=40% height=40%><br/>(b)<br/><img src="https://user-images.githubusercontent.com/102669387/217750101-fd94ed7a-d281-45b1-94da-ab00b351c6e2.png" width=40% height=40%><br/>(c)<br/>Figure 1. Illustration of time-serial data with data preprocessing. (a) The filtered time-serial data $\tilde{X}$. (b) The time-serial data with zero padding $\hat{X}$. (c) The corresponding class label sequence $C$ and the set of transition point label $T$ for $\hat{X}$.</p>

## Deep MTL U-Net
<p align="center"><img src="https://user-images.githubusercontent.com/102669387/209524513-60931bc6-7683-4b14-80e5-259615606ff8.png" width=80% height=80%></p>

The figure above presents the architecture of the proposed deep MTL U-Net. The structure can be separated into three parts: the STS encoder 𝐺𝑒, the STS decoder 𝐺𝑑, and the transition point detector 𝐺𝑡. 𝐺𝑒 and 𝐺𝑑 perform sub-task classification on each time point for the STS task while 𝐺𝑒 and 𝐺𝑡 perform the TPD task. Both tasks share the parameters of 𝐺𝑒.

## Experimental results  

## A simple guideline for practice
Besides source codes, this repostiry provides a [trained deep MTL U-Net](https://drive.google.com/file/d/10R9mnqxuRENmgr3JhNi1pg9OOqXd_-IR/view?usp=share_link), a [validation set](/val_set.npy), and a [validation script](/validation.py) for demonstration.


## Disclaimer
This is still a work in progress and is far from being perfect: if you find any bug please don't hesitate to open an issue.
