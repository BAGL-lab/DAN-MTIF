# Deep Attention Networks with Multi-Temporal Information Fusion for Sleep Apnea Detection

## Abstract
Sleep Apnea (SA) is a prevalent sleep disorder with multifaceted etiologies that can have severe consequences for patients. Diagnosing SA conventionally relies on the in-laboratory polysomnogram (PSG). PSG  records various human physiological activities overnight, and SA diagnosis involves manual scoring by qualified physicians. Traditional machine learning methods for SA detection depend on hand-crafted features, making feature selection pivotal for downstream classification tasks. In recent years, deep learning has gained popularity in SA detection due to its automatic feature extraction and superior classification accuracy. This study introduces a Deep Attention Network with Multi-Temporal Information Fusion (DAN-MTIF) for SA detection using single-lead electrocardiogram (ECG) signals. This framework utilizes three 1D convolutional neural network (CNN) blocks to extract features from R-R intervals and R-peak amplitudes using segments of varying lengths. Recognizing that features derived from different temporal scales vary in their contribution to classification, we integrate a multi-head attention module with a self-attention mechanism to learn the weights for each feature vector. Comprehensive experiments and comparisons between two paradigms of classical machine learning approaches and deep learning approaches are conducted. Our experiment results can demonstrate that (1) compared with benchmark methods, the proposed DAN-MTIF exhibits excellent performance with 0.9106 accuracy, 0.9396 precision, 0.8470 sensitivity, 0.9588 specificity, 0.8909 $F_1$ at per-segment level; (2) DAN-MTIF can effectively extract features with a higher degree of discrimination from ECG segments of multiple timescales than those with a single time scale, which guarantees a better SA detection performance; (3) The overall performance of deep learning approaches is better than the classical machine learning algorithms, highlighting the superior performance of deep learning for SA detection. 


## Dataset
[Apnea-ECG Database](https://physionet.org/content/apnea-ecg/1.0.0/)


## Cite
If our work is helpful to you, please cite:

## Email
If you have any questions, please email to: [fliu22@stevens.edu](mailto:fliu22@stevens.edu)
