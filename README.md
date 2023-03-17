# FLITC-application
A data-driven deep learning-based Fault Location Identification and Type Classification application for radial and active distribution grids. 
It also contains the source code for dataset generation for hyperparameter tuning, training, validation and testing of the models.   



Please cite our journal paper as such:

V. Rizeakos, A. Bachoumis, N. Andriopoulos, M. Birbas, A. Birbas,
Deep learning-based application for fault location identification and type classification in active distribution grids,
Applied Energy,
Volume 338,
2023,
120932,
ISSN 0306-2619,
https://doi.org/10.1016/j.apenergy.2023.120932.
(https://www.sciencedirect.com/science/article/pii/S0306261923002969)


Abstract: The high penetration of distributed energy resources, especially weather-dependent sources, even at the edge of the distribution grids, has increased the power system uncertainties and drastically shifted the operational status quo for the system operators. For the operators to ensure the uninterrupted electricity supply of the end-consumers, the fast and accurate response to fault events is of critical importance. This paper proposes a data-driven fault location identification and types classification application based on the continuous wavelet transformation and convolutional neural networks optimally configured through Bayesian optimization. This application leverages the proliferation of high-resolution measurement devices in distribution networks. It can locate the exact position of the short-circuit faults and classify them into eleven different types. Its intrinsic models grasp the spatial characteristics and the converted in frequency domain temporal ones of the three-phase voltage and current timeseries measurements stemming from the field devices, thus increasing the operators’ visibility of their networks in real-time. We conduct simulations through synthetic data, which we provide in an open-source repository, that replicate a wide range of fault occurrence scenarios with eleven different types, with the resistance ranging from 50Ω to 2kΩ and with duration from 20ms to approximately 2s, under noise conditions injected by devices and load variability. The results showcase the efficacy of the proposed method reaching an accuracy of 91.4% for fault detection, 93.77% for correct branch identification, 94.93% for fault type classification, and RMSE value of 2.45% for location calculation.
Keywords: Active distribution grids; CNNs; Deep learning; Fault detection and location identification; Wavelet transformation

