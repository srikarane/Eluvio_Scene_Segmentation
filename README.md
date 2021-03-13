# Eluvio_Scene_Segmentation
Predicting the scene segmentation for each movie given features for each shot.

## Author: Srikaran Elakurthy
**Requirements:**
- Python
- Pytorch
- Tensorflow
- Sckitlearn
- TPU
- Google Colab or Jupyter Notebook

**Raw Data Folder:**
To work with the project, Raw data and splitted data(Train and test data are stored in: https://drive.google.com/drive/folders/1au8EnqzlpRw-8FPJli2a22yIcwHw-rqO?usp=sharing ) must be obtained from the link provided in their respective folders. Before proceeding with the project make sure you have these files downloaded in the root folder of the project.

**Source folder in every program is the root folder of the project**

**Reason For choosing Google colab:**
- Due to resource exhaustion in the machine, not able to flexibly perform transformations and build model, so choosed google colab to leverage the TPU resources. 

### FINAL MODEL and its Metrics:
Final Model is choosed was the Scenario 1 where we achieved Miou as **0.421** and with a decent Recall but recall was good for Scenario 3 but making Miou as main metric choosed Scenario1 as final model.
**Metric*:**
"AP": 0.27538717329839113,

"mAP": 0.30240989906638455,

"Miou": 0.42189183857398244,

"Precision": 0.2486060100297133,

"Recall": 0.581917462249597,

"F1": 0.338406696537817

### What can be done more:
- If we have the whole dataset of 158 movies as in the paper, the metrics have improved more with provided resources by leveraging the distribution strategies of TPU and GPU.
- Usage of shot end frames, which can be used to know the length of the shot believing it may become one of the crucial feature in deciding scene boundaries.
- Once the predictions are done we can even perform Non integer programming and find the parameters for minimum loss.

### Description:
Analyzing the shot information of different movies and predict whether a shot is a scene boundary or not. Prediction has been tried on 3 different scenario and the final and selected scenario is in Final folder. In all the scenarios we have used window implementation because information of before and after shot are crucial in deciding the scene boundary. We will see in this repository the different ways of usage of this information effectively.

#### Splitting the dataset:
We divided the dataset into train and test set of 80% and 20% respectively consisting of 54 and 12 movies respectively. Validatio split 20% which is divided at runtime of the model.

#### Chanllenges:
 - Needed huge computation power in normal machine so shifted to Google colab and used TPU and High Ram.
 - Class Imbalance, to mitigate this issue we used Binary Focal Loss as loss function.
#### Scenario Descriptions:
Following Scenarios has been implemented 
1. **Scenario 1:**
    - Choosen the even window size so that we can divide the window into half and make them as channels. So that we can capture the equal amount of relationships and differences from both sides of the shots. If we have shots {s<sub>1</sub>,s<sub>2</sub>,s<sub>3</sub>,....s<sub>n</sub>) and window size as 6 then for shot s<sub>i</sub> we will have first channel as (s<sub>i-2</sub>,s<sub>i-1</sub>,s<sub>i</sub>) and second channel as (s<sub>i+1</sub>,s<sub>i+2</sub>,s<sub>i+3</sub>).
    - So that the same amount of convolution happens in both sides by channeling them.
    - We used Convolution 2d to capture the semantic relationships on both sides and by Maxpooling we concentrate on extracting the important information which we can then finally use in Dense layer and analyze them to get the predictions of Scene boundary.
    - After Hypertuning with a varied window sizes, window size 8 has given better results.
    - This is a Final model which has the Miou of **0.421**
    - Model and architecture can be found in the final folder
2. **Scenario 2:** 
    - Choosen the Odd window so that there would be equal number of shots on both sides of a shot keeping the selected shot in the middle and vertically stack them. So, for example if window size is 5 Then for a particular shot say shot number 5 so this shot will have the features of shot 3,4 and 6,7 which we stack. So, for each shot we will have (5,3584) tensor where 3584 is features combined of place,cast,action and audio.
    - By using Convolutions of 1 dimension we capture the clip level information and by using LSTM's and a Dense layer with Sigmoid Functionality we will predict the Scene Boundary.
    - After Hypertuning with a varied window sizes, window size 7 has given better results
    - The Miou best achieved was **0.415**
    - Model and architecture can be found in shot_level_win_odd folder.
3. **Scenario 3:**
    -   Choosen the even window size just as the first scenario but instead we wont split them as channel but conversly we vertically stack them.
    -   Divided the Convolutions in two branches in Branch 1 we first split each shot exactly to half consiting of shots B1{(s<sub>i-2</sub>,s<sub>i-1</sub>,s<sub>i</sub>),(s<sub>i+1</sub>,s<sub>i+2</sub>,s<sub>i+3</sub>)} and convolve them by 2 convolution layers following a dot product aimimg to capture the differences and in other branch B2(s<sub>i-2</sub>,s<sub>i-1</sub>,s<sub>i</sub>,s<sub>i+1</sub>,s<sub>i+2</sub>,s<sub>i+3</sub>) a convolution layer followed by a Maxpool layer we capture the relations.
    -   Then we concatenate them and will send to LSTM and dense layers to coarse the predictions of scene boundary.
    -   After Hypertuning with a varied window sizes, window size 8 has given better results.
    -   The Miou best achieved was **0.37**
    -   The model and architechture can be found in the Shot_split folder.
 
