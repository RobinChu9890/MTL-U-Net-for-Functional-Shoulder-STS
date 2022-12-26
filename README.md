# Multi-Task Learning U-Net for Functional Shoulder Sub-Task Segmentation

In functional shoulder assessment, functional shoulder sub-tasks could provide more function information for clinical frozen shoulder (FS) assessment. However, label annotation 
for shoulder sub-tasks still relies on manual observation and operation in medical practice, which is time-consuming and prone to errors. To support clinical evaluation, this work 
proposes a deep multi-task learning (MTL) U-Net for automatic functional shoulder sub-task segmentation (STS). The transition point detection (TPD) based on convolutional 
neural networks (CNN) serves as the auxiliary task during the training stage. The fine-grained transition-related information from TPD task helps STS task have better ability 
to tackle the boundary between functional shoulder sub-tasks, and TPD task obtains critical contextual knowledge from STS task to precisely detect transition points between shoulder
sub-tasks. MTL transfers the knowledge across tasks and boost the performance of STS. We conduct the experiments using wearable inertial measurement units to record 815 
shoulder task sequences, which is collected from 20 healthy subjects and 43 patients with FS. The experimental results present that the deep MTL U-Net can achieve superior 
performance compared to using single-task models. It shows the effectiveness of the proposed method for functional shoulder STS.
