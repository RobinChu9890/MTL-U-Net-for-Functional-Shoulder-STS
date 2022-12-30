# Multi-Task Learning U-Net for Functional Shoulder Sub-Task Segmentation

A trained deep MTL U-Net could be downloaded here:
https://drive.google.com/file/d/10R9mnqxuRENmgr3JhNi1pg9OOqXd_-IR/view?usp=share_link

## Abstract
In functional shoulder assessment, functional shoulder sub-tasks could provide more function information for clinical frozen shoulder (FS) assessment. However, label annotation for shoulder sub-tasks still relies on manual observation and operation in medical practice, which is time-consuming and prone to errors. To support clinical evaluation, this work proposes a deep multi-task learning (MTL) U-Net for automatic functional shoulder sub-task segmentation (STS). The transition point detection (TPD) based on convolutional neural networks (CNN) serves as the auxiliary task during the training stage. The fine-grained transition-related information from TPD task helps STS task have better ability to tackle the boundary between functional shoulder sub-tasks, and TPD task obtains critical contextual knowledge from STS task to precisely detect transition points between shoulder sub-tasks. MTL transfers the knowledge across tasks and boost the performance of STS.  
  
In this repostiry, we would breifly introduce the structure of the proposed deep MTL U-Net and the collected dataset. To present the effectiveness of our network, we conduct the experiments using wearable inertial measurement units (IMUs) to record 805 shoulder task sequences, which is collected from 20 healthy subjects and 43 patients with FS. The dataset is splited into around 80% (655 sequences) and 20% (150 sequences) for training and validation, respectively. The trained network and the validation set would be provided in this repostiry.  

## An overview of the proposed deep MTL U-Net
<p align="center">
<img src="https://user-images.githubusercontent.com/102669387/209524513-60931bc6-7683-4b14-80e5-259615606ff8.png" width=80% height=80%>

The figure above presents the architecture of the proposed deep MTL U-Net. The structure can be separated into three parts: the STS encoder 𝐺𝑒, the STS decoder 𝐺𝑑, and the transition point detector 𝐺𝑡. 𝐺𝑒 and 𝐺𝑑 perform sub-task classification on each time point for the STS task while 𝐺𝑒 and 𝐺𝑡 perform the TPD task. Both tasks share the parameters of 𝐺𝑒.

## The experimental validation set
This repostiry provides the validation set containing 163 time sequences of functional shoulder tasks collected with IMUs. 
  * 20 subjects: 10 healthy subjects and 10 patients with FS
  * 5 functional shoulder tasks: washing head (WH), washing upper back (WUB), washing lower back (WLB), putting an object on a high shelf (POH), and removing an object from the back pocket (ROB)
    * Each task is performed once in one recording session and is divided into three shoulder sub-tasks.
    * The shoulder sub-task description of five selected shoulder tasks are shown in the table below.
    <p align="center">
    <img src="https://user-images.githubusercontent.com/102669387/209618125-c054a3cb-7312-456b-a6da-97efd882ca6f.png"  width=35% height=35%>
      
    * Sub-task 1, 2, and 3 of different tasks are trained as the same class to validate the generality of the proposed method.  
  * 2 IMUs (APDM Inc., Portland, USA)
    * sampling rate of 128 Hz
    * Fastened to the wrist and upper arm of the dominant side for healthy subjects and the affected side for patients.
    * Each IMU contains a tri-axial accelerometer (range: ±16 g, resolution: 14 bits) and a tri-axial gyroscope (range: ±2000 °/s, resolution: 16 bits) to collect time-serial data with 4 modalities and 3 axes.  

> All tasks sequences we collected are applied with zero-padding for length resizing. This resizing process ensures all sequences of the dataset to the same size, 𝑙𝑚𝑎𝑥. The added zero samples are labeled as a new sub-task class to be distinguished from the original shoulder sub-task samples.  
> The sub-task boundaries in each IMU sequence are normalized with a respect to 𝑙𝑚𝑎𝑥.

## Experimental results
<p align="center">
<img src=""  width=80% height=80%>

### Disclaimer
This is still a work in progress and is far from being perfect: if you find any bug please don't hesitate to open an issue.
