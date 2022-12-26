# Multi-Task Learning U-Net for Functional Shoulder Sub-Task Segmentation

## Abstract
In functional shoulder assessment, functional shoulder sub-tasks could provide more function information for clinical frozen shoulder (FS) assessment. However, label annotation 
for shoulder sub-tasks still relies on manual observation and operation in medical practice, which is time-consuming and prone to errors. To support clinical evaluation, this work 
proposes a deep multi-task learning (MTL) U-Net for automatic functional shoulder sub-task segmentation (STS). The transition point detection (TPD) based on convolutional 
neural networks (CNN) serves as the auxiliary task during the training stage. The fine-grained transition-related information from TPD task helps STS task have better ability 
to tackle the boundary between functional shoulder sub-tasks, and TPD task obtains critical contextual knowledge from STS task to precisely detect transition points between shoulder
sub-tasks. MTL transfers the knowledge across tasks and boost the performance of STS. We conduct the experiments using wearable inertial measurement units to record 815 
shoulder task sequences, which is collected from 20 healthy subjects and 43 patients with FS. The experimental results present that the deep MTL U-Net can achieve superior 
performance compared to using single-task models. It shows the effectiveness of the proposed method for functional shoulder STS.


## An overview of the proposed deep MTL U-Net
![image](https://user-images.githubusercontent.com/102669387/209524513-60931bc6-7683-4b14-80e5-259615606ff8.png)
The figure above presents the architecture of the proposed deep MTL U-Net. The structure can be separated into three parts: the STS encoder 𝐺𝑒, the STS decoder 𝐺𝑑, and the transition point detector 𝐺𝑡. 𝐺𝑒 and 𝐺𝑑 perform sub-task classification on each time point for the STS task while 𝐺𝑒 and 𝐺𝑡 perform the TPD task. Both tasks share the parameters of 𝐺𝑒.

## Results
![image](https://user-images.githubusercontent.com/102669387/209527998-49b88213-7ea7-4d14-9b19-e3a5495b12c4.png)

### Disclaimer
This is still a work in progress and is far from being perfect: if you find any bug please don't hesitate to open an issue.
