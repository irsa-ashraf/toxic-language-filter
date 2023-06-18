# Toxic Language Filter
Final Project for the class CAPP30255: Advanced Machine Learning for Public Policy, taught by Professor Amitabh Chaudhry at the University of Chicago.

## Project Overview
Our goal is to build a classifier that can accurately classify and categorize toxic comments on social media platforms. The model will be trained on a pre-labeled dataset containing toxic and non-toxic online comments to identify different levels of toxicity on media platforms.

At the most fundamental level, the project aims to develop an NLP model that can accurately perform binary classification between toxic and non-toxic comments. In addition, the project aims to identify different categories of toxicity, such as identity-based hate, threats, insults, and others. Ultimately, the project sets the goal to develop a generalized, scalable language model as a solution to classifying different types of toxic comments across different platforms. This required training on different datasets, fine-tuning the model for high accuracy, and performance optimization–some of it was out of the scope of the course and so we adjusted the implementation according to time and resources available to us.


### Data Source
Jigsaw Toxic Comment Classification Dataset from kaggle 

The dataset consists of a large number of comments from Wikipedia’s talk page edits, along with binary labels indicating whether each comment is toxic or not. The data is in a CSV format, with each row representing a single comment and its associated labels. The dataset contains around 310,000 comments, and each comment is labeled on 6 different types of toxicity: toxic, severe toxic, obscene, threat, insult, and identity hate. One comment can be categorized as more than one type of toxicity.

### Project Contributors:
- [Irsa Ashraf](https://github.com/irsa-ashraf)
- [Yifu Hou](https://github.com/yifu-hou)
- [Ken Kliesner](https://github.com/kenkliesner)


