# Multi-Task Learning U-Net for Functional Shoulder Sub-Task Segmentation
The assessment of a frozen shoulder (FS) is critical for evaluating outcomes and medical treatment. Analysis of functional shoulder sub-tasks provides more crucial information, but current manual labeling methods are time-consuming and prone to errors. To address this challenge, we propose a deep multi-task learning (MTL) U-Net to provide an automatic and reliable functional shoulder sub-task segmentation (STS) tool for clinical evaluation in FS. The proposed approach contains the main task of STS and the auxiliary task of transition point detection (TPD). For the main STS task, a U-Net architecture including an encoder-decoder with skip connection is presented to perform shoulder sub-task classification for each time point. The auxiliary TPD task uses lightweight convolutional neural networks architecture to detect the boundary between shoulder sub-tasks. A shared structure is implemented between two tasks and their objective functions of them are optimized jointly. The fine-grained transition-related information from the auxiliary TPD task is expected to help the main STS task better detect boundaries between functional shoulder sub-tasks. We conduct the experiments using wearable inertial measurement units to record 815 shoulder task sequences collected from 20 healthy subjects and 43 patients with FS. The experimental results present that the deep MTL U-Net can achieve superior performance compared to using single-task models. It shows the effectiveness of the proposed method for functional shoulder STS.



## Data preprocessing
Before fed into the network, all time-serial data is first denoised with simple moving average (SMA) filter by averaging a group of samples. A 5-point SMA filter is calculated as the equation below:

$$\tilde{x}_n = \frac{x_{n-2}+x_{n-1}+x_n+x_{n+1}+x_{n+2}}{5},$$

where $\tilde{x}$ is the filtered sample, $x$ is the original sample, and $n$ is the sample index. The example smoothed time-serial data is presented in Fig. 1 (a).

Next, we apply zero-padding to each filtered time-serial sequence $\tilde{X}$ for length resizing. This resizing process ensures all sequences to the same size. Let $l_i$ be the length of sequence $i$, and $l_{max}$ is the maximum of $\\{l_i│\forall i \in [1,n]\\}$, where $n$ is the total number of sequences. Zero values are added before and after each original time-serial sequence to ensure the new sequence $\hat{X}$ have the same length equal to $l_{max}$, as shown in Fig. 1 (b).<br/>
The added zero samples are labeled as a new sub-task class to be distinguished from the original shoulder sub-task samples. The sub-task boundaries in each IMU sequence are normalized with a respect to $l_{max}$. The resized sequence of class labels $C$ and the set of transition points $P$ for $\hat{X}$ are illustrated in Fig. 1 (c).

<p align="center"><img src="https://user-images.githubusercontent.com/102669387/217750074-4cbdceb9-d3e7-48a0-ae34-a58807a9cc1a.png" width=40% height=40%><br/>(a)<br/><img src="https://user-images.githubusercontent.com/102669387/217750086-a319831c-6619-4105-8f8f-2121b19d6895.png" width=40% height=40%><br/>(b)<br/><img src="https://user-images.githubusercontent.com/102669387/217750101-fd94ed7a-d281-45b1-94da-ab00b351c6e2.png" width=40% height=40%><br/>(c)<br/>Figure 1. Illustration of time-serial data with data preprocessing. (a) The filtered time-serial data $\tilde{X}$. (b) The time-serial data with zero padding $\hat{X}$. (c) The corresponding class label sequence $C$ and the set of transition point label $T$ for $\hat{X}$.</p>

## Deep MTL U-Net
<p align="center"><img src="https://user-images.githubusercontent.com/102669387/209524513-60931bc6-7683-4b14-80e5-259615606ff8.png" width=80% height=80%><br/>Figure 2. The architecture of proposed deep MTL U-Net</p>

Fig. 2 presents the architecture of the proposed deep MTL U-Net. The structure can be separated into three parts: the STS encoder $G_e$, the STS decoder $G_d$, and the transition point detector $G_t$. $G_e$ and $G_d$ perform sub-task classification on each time point for the STS task while $G_e$ and $G_t$ perform the TPD task. Both tasks share the parameters of $G_e$.
* $G_e$ contains recurrent union of two convolutional layers with kernel size of 1-by-3 and one max-pooling layer with kernel size as 1-by-2. The number of convolutional kernels is doubled after each max-pooling layer. Padding as 1 and stride as 1 are set for convolutional layers to maintain the sequence length. The final contextual encoding is next passed to $G_d$ and $G_t$ respectively.
* $G_d$ has a symmetry structure of the encoder, but max-pooling layers are replaced with up-convolutional layer having kernel size as 1-by-2 to increase length and cut channel number in half. After up-convolution, the feature is concatenated with the sequence from corresponding encoder layer to conserve extracted spatial characteristics. For the last layer, a convolutional layer with kernel size as 1-by-1 is used for mapping the feature sequences to the class number.
* $G_t$ uses multilayer perceptron (MLP) as the main component. We flatten the contextual encoding from the output of $G_e$, and then input them to two fully connected layers with 210 and 4 neurons.

Given an IMU sequence with zero padding $\hat{X} = \\{\hat{x}^{m,a}\_i │ \forall i \in [1,l\_{max}], m \in M, a \in A\\}$, where $\hat{x}\_i$ is the sample point at time point $i$, $M$ is the modality set, and $A$ is the axis set. The corresponding class label sequence $C = \\{c_i | \forall c_i \in L, i \in [1,l_{max}]\\}$ and a transition points set $P = \\{p_j | \forall p_j \in [1,l_{max}], j \in [1,n_p]\\}$ are determined as the target for STS and TPD respectively, where $L$ is the sub-task class set, and $n_p$ is the number of sub-task boundary. The predicted class label sequence $\hat{C} = \\{\hat{c}\_i | \forall \hat{c}\_i \in L, i \in [1,l_{max}]\\}$ from $G_d$ and the predicted set of transition points $\hat{P} = \\{\hat{p}\_j | \forall \hat{p}\_j \in [1,l_{max}], j \in [1,n_p]\\}$ from $G_t$ are given formally by the following equations:

$$\hat{C} = G_d(G_e (\hat{X}; θ_e ); θ_d),$$

$$\hat{P} = G_t(G_e (\hat{X}; θ_e ); θ_t),$$

where $θ_e$, $θ_d$, and $θ_t$ are the parameters of $G_e$, $G_d$, and $G_t$, respectively.

## Data collection
This study collects a dataset containing time sequences of functional shoulder tasks collected from IMUs. We recruit 63 subjects, including 20 healthy subjects (10 males, 17 right-handedness, age: 24.55±3.76 years old, height: 168.60±6.73 cm, weight: 67.95±15.34 kg) and 43 patients with FS (16 males, 18 right side affected, 7 both sides affected, age: 57.63±10.58 years old, height: 171.77±47.91 cm, weight: 63.10±11.38 kg). Five functional shoulder tasks are selected from the Shoulder Pain and Disability Index (SPADI) questionnaire[^1], containing washing head (WH), washing upper back (WUB), washing lower back (WLB), putting an object on a high shelf (POH), and removing an object from the back pocket (ROB). The data collection is approved by the institutional review board (TSGHIRB No.: A202005024) at the university hospital. All subjects are provided informed consent and voluntary for participation.

[^1]: J. D. Breckenridge and J. H. McAuley, "Shoulder pain and disability index (SPADI)," Journal of physiotherapy, vol. 57, no. 3, pp. 197-197, 2011.

Each task is performed once in one recording session and is divided into three shoulder sub-tasks. A total 815 shoulder task sequences are recorded in this study, where 100 sequences performed by healthy subjects and 143 sequences performed by patients at their first and follow-up visits. The longest sequence length l_max is 3798. The shoulder sub-task description of five selected shoulder tasks are shown in Table 1. Sub-task 1, 2, and 3 of different tasks are trained as the same class to validate the generality of the proposed method. Two IMUs (APDM Inc., Portland, USA) with sampling rate of 128 Hz are fastened to the wrist and upper arm of the dominant side for healthy subjects and the affected side for patients. Each IMU contains a tri-axial accelerometer (range: ±16 g, resolution: 14 bits) and a tri-axial gyroscope (range: ±2000 °/s, resolution: 16 bits) to collect time-serial data with 4 modalities and 3 axes.

<p align="center">Table 1. Shoulder sub-task description of five shoulder tasks</p>
<table align="center">
  <tr><th>Task</th><th>Sub-task</th><th>Description</th></tr>
  <tr><td rowspan="3" align='center'>WH</td><td align='center'>1</td><td>Lift up both hands toward the head</td></tr>
    <tr><td align='center'>2</td><td>Wash head for a few seconds</td></tr>
    <tr><td align='center'>3</td><td>Put down both hands and return to the initial position</td></tr>
  <tr><td rowspan="3" align='center'>WUB</td><td align='center'>1</td><td>LLift up the dominant / affected hand toward the upper of back</td></tr>
    <tr><td align='center'>2</td><td>Wash upper back for a few seconds</td></tr>
    <tr><td align='center'>3</td><td>Put down the dominant / affected hand and return to the initial position</td></tr>
  <tr><td rowspan="3" align='center'>WLB</td><td align='center'>1</td><td>Lift up the dominant / affected hand toward the lower of back</td></tr>
    <tr><td align='center'>2</td><td>Wash lower back for a few seconds</td></tr>
    <tr><td align='center'>3</td><td>Put down the dominant / affected hand and return to the initial position</td></tr>
  <tr><td rowspan="3" align='center'>POH</td><td align='center'>1</td><td>Lift up the dominant / affected hand toward a high shelf while holding a smartphone</td></tr>
    <tr><td align='center'>2</td><td>Hold the hand for a few seconds</td></tr>
    <tr><td align='center'>3</td><td>Put down the dominant / affected hand and return to the initial position</td></tr>
  <tr><td rowspan="3" align='center'>ROB</td><td align='center'>1</td><td>Putting a smartphone from the initial position to the back pocket with the dominant / affected hand</td></tr>
    <tr><td align='center'>2</td><td>Hold the hand for a few seconds</td></tr>
    <tr><td align='center'>3</td><td>Removing the smartphone from the back pocket to the initial position with the dominant/affected hand</td></tr>
</table>

## Implementation details
The element number of sub-task class set $L$ is four, including three functional shoulder sub-task and one zero-padding class. The number of sub-task boundary $n_p$ is four. The optimizer is AdamW with an initial learning rate of 0.001. A total 128 epochs are used for mini-batch training, where the batch size is 64.<br/>
This work utilizes 10-fold cross validation on the collected dataset for performance evaluation. Three common metrics are chosen as the criteria for performance evaluation, including recall, precision, and F1-score.<br/>
The experiments are processed and examined on python 3.9 in a Windows 11 environment with a GPU of NVIDIA RTX 3080. The deep learning network is programed using PyTorch 1.12.1 with CUDA 11.6.

## Experimental results
To demonstrate the effectiveness of the proposed deep MTL U-Net for STS, we compare the proposed method with baseline models without MTL, including a single-task U-Net for STS and a single-task CNN for TPD. These simplified networks have the same experimental details, and their parameters are optimized. The experiment results are presented in Table 2. It shows that the proposed deep MTL U-Net can reach superior performance to the single-task models. The F1-score on STS and TPD are increased by 0.66% and 0.32%, respectively. Moreover, the segmentation f1-score (89.92%) of the proposed model notably outperforms that (83.23%) of the prior study[^2] approach using conventional sliding window and machine learning techniques.

[^2]: C.-Y. Chang et al., "Automatic functional shoulder task identification and sub-task segmentation using wearable inertial measurement units for frozen shoulder assessment," Sensors, vol. 21, no. 1, p. 106, 2020.

<p align="center">Table 2. The performance comparison between deep MTL U-Net and simplified models<br/></p>
<table align="center">
  <tr><th>Task</th><th>Structure</th><th>Recall (%)</th><th>Precision (%)</th><th>F1-score (%)</th></tr>
  <tr><td rowspan="2" align='center'>STS</td><td align='center'>Deep MTL U-Net</td><td align='center'><b>90.31</b></td><td align='center'><b>89.64</b></td><td align='center'><b>89.92</b></td></tr>
    <tr><td align='center'>U-Net only (without $G_T$)</td><td align='center'>89.52</td><td align='center'>89.08</td><td align='center'>89.26</td></tr>
  <tr><td rowspan="2" align='center'>TPD</td><td align='center'>Deep MTL U-Net</td><td align='center'>88.26</td><td align='center'>87.61</td><td align='center'>87.56</td></tr>
    <tr><td align='center'>CNN only (without $G_D$)</td><td align='center'>87.62</td><td align='center'>87.55</td><td align='center'>87.56</td></tr>
</table>

## A simple guideline to practice
Besides source codes, this repostiry provides a [trained deep MTL U-Net](https://drive.google.com/file/d/10R9mnqxuRENmgr3JhNi1pg9OOqXd_-IR/view?usp=share_link), a [validation set](/val_set.npy), and a [validation script](/validation.py) for demonstration.
* The network is trained with 655 time sequences (around 80%) of the collected functional shoulder tasks.
* The validation set contains the remainging data, which comprises 150 time sequences collected from 2 healthy subjects and 4 patients with FS.
* The validation script is a well coded program that would automatically retrieve the network and data and present the validation results as figures.
To make the script functioning well, please ensure that 

## Disclaimer
This is still a work in progress and is far from being perfect: if you find any bug please don't hesitate to open an issue.
