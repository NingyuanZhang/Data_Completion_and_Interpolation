# Data_Completion_and_Interpolation
A system for predicting or interpolating missing features from the present features in new records.
## Dataset
The dataset is a set of psychological questionnaires. Each row represents a participant, and each column represents a question.
## Data selection and preprocessing
Due to the time limitation, we only consider to do the data completion and interpolation on the features in the variable codebook,  
which are more important and valuable for analysis. 

Note that there is an attention check in the ML3 dataset. If the answer of this question is not
“I read the instructions”, it represents that the user gave the answers to the questionnaire without
reading the questions first, which means that their data is just the noise and will definitely provide
bad influence to our model. Thus, we firstly check the content of this feature “attention correct”
and drop out all the noise in the dataset. 

There are 35 features with type “effect” and 46 features with type
“individual difference”. In our project, we choose to use the features with type “effect” as our
features which need to be predicted or reconstructed from the features that are present and the
features with type “individual difference” as our latent relevant features. 

## Model selection
### Decision Tree Model
### Cross Entropy Model
