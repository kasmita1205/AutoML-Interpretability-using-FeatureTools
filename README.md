# AutoML-Interpretability-using-FeatureTools
#### -Asmita Khode
#### Guided by- Dr. Nigel Bosch

Applying tools such as FeatureTools to automatically extract features from Open University Learning and Analytics Dataset to understand and interpret important features which contribute to final outcomes of students performance in assessments and use these features for developing Machine Learning models which can predict the outcome for a given student. 

## Contents
- Introduction
- WorkFlow
- Challenges
- Findings and Learnings
- Future Work

### Introduction
For any machine learning technique, feature extraction that yields high model accuracy has always been a tough process. Although developments of  recent AutoML tools such as Tsfresh and FeatureTools have made it easier to extract thousands of features with a little effort which has helped in improving model accuracy significantly. But the major challenge with these AutoML engineered features is interpretability, for both experts and novices. Hence, in this project we aim to understand these interpretability issues and work on extracting features which are more interpretable  and also studying people’s ability to interpret models based on these features.

### WorkFlow
#### Dataset
For the purpose of this project we focused on exploring the ![Open University Learning Analytics dataset (OULAD)](https://archive.ics.uci.edu/ml/datasets/Open+University+Learning+Analytics+dataset?msclkid=ff74068cd07f11ec82ffaf6c8f51144f). This dataset contains data from a Virtual Learning Environment  which has seven different modules. The behaviour or the interaction of students on these different modules while using Virtual Learning Environment is recorded and also, the details of assessment and final grade is recorded for analysis.

#### Data Preparation and Processing
For feeding our data into FeatureTools we have done some data preparation. We have combined the code_module and code_presentation as “courseKey” from the course table to make it the primary key .  We have also made this courseKey in all other relevant tables as well so that the tables can be joined together. As FeatureTools takes data in the form of Entityset which maintains details of all the relevant tables and relationship between them we added all our tables as entities into an entityset and also defined relationships between them which is nothing but primary key and foreign key relations. We have also dropped the final_result column from our studentInfo table as it is a target entity. To test our feature extraction with limited memory resources we have used just 20% of our records from the VLE table i.e studentVle.

#### Feature Extraction(FeatureTools)
For the purpose of feature extraction using FeatureTools we have used aggregation primitives like ‘max’, ‘median’,’mode’, ‘sum’ etc and transform primitives such as ’percentile’, ‘cum_sum’, ‘cum_max’ and many more. Here we have chosen our target entity for extracting our features as a “resultant” table  which has a unique student id associated with each record and their respective final_result. We are also in the process of making a complete_id key which is a combination of student_id and courseKey for better result and accuracy.

#### Modelling
For modelling our data on extracted features we have considered Random Forest Classifier for now because our outcome or response variable i.e final_result is divided into five different categories- Pass, Fail, Withdraw, Distinction and Unknown respectively. At this stage we are trying to design single feature models to evaluate performance of each feature individually. 

### Challenges
- Dealing with all records altogether in studentVle table for feature extraction and modelling due to limited memory resources.
- Applying all aggregation and transformation primitives in FeatureTools to extract features is a challenge as it generates a huge set of features which is difficult to deal with available processor and memory.

### Finding and Learnings
AutoML tools are a very efficient way to extract a large number of features in comparatively less time. It saves a lot of human effort. Although it requires some manipulation of data to be fed in these tools, with better understanding of data and these tools the time and effort required are really less than manual feature extraction. While working on this project we explored two AutoML tools- Tsfresh and FeatureTools. We learnt how to use these AutoML tools to extract and store features. Further we have also learnt how we can use these stored extracted features in our Machine Learning models.

### Future Work
The future scope of this project is to extract more features using completeKey and compare the results obtained from the previous approach. After extracting features we look forward to trying different machine learning models on various different features and look for model performance using Cohen’s score. Finally we will dive deeper into interpreting the features which perform best on our models.
