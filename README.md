Fatigue Level Detection using Heart Rate Sensor

This project aims to build a system that can detect fatigue levels caused by prolonged and repetitive use of computer mice. The system utilizes a heart rate sensor to collect heart rate measurements, which are then used to train an artificial neural network (ANN) model for fatigue level prediction.

Overview:

Overusing a computer mouse can lead to muscle soreness and fatigue due to the repetitive actions of small muscle groups. This project addresses this issue by developing a system that can identify the level of stress and fatigue experienced by the user.

System Architecture:

The system architecture involves the following components:

1. Data Collection: Heart rate data is collected using a heart rate sensor while the user is operating a computer mouse.
2. Data Preprocessing: The collected data is preprocessed, including noise filtering and segmentation into 30-second frames.
3. Data Labeling: The preprocessed data frames are labeled as either "normal" or "stressed" based on the user's reported fatigue levels.
4. Model Training: An artificial neural network (ANN) model is trained using the labeled data frames.
5. Model Evaluation: The trained model is evaluated on a test dataset to determine its accuracy in predicting fatigue levels.

Experimental Evaluation:

- Data was collected for 4 hours of active mouse usage.
- The collected data was preprocessed, filtered for noise, and segmented into 30-second frames.
- The data frames were labeled as "normal" or "stressed" based on the user's reported fatigue levels.
- The labeled data was used to train an ANN model using TensorFlow.
- The trained model achieved an accuracy of 78.6% in predicting fatigue levels based on heart rate data.

Conclusion:

This project demonstrates the feasibility of using heart rate data and an ANN model to detect fatigue levels caused by prolonged computer mouse usage. While the achieved accuracy of 78.6% is promising, further improvements in data preprocessing and model training may be required to enhance the system's performance.
