# CNN Language Model for Identifying Toxic Languages from Online Comments

## About this project

The project aims to build a language model that can accurately classify and categorize toxic comments on social media platforms. The model will be trained on a pre-labeled dataset containing toxic and non-toxic online comments to identify different levels of toxic on media platforms. </p>

In addition, the project aims to identify different categories of toxicity, such as identity-based hate, threats, insults, and others. </p>

The baseline model for this project is a simple Bag-Of-Words (BoW) model, and the main model is a Convolutional Neural Network (CNN). Both models use `GloVe` for word embedding.

## Dataset 

The training dataset for this project is Jigsaw Toxic Comment Classification Challenge dataset on Kaggle ([link](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data)). </p>

The average length of any individual comment’s raw text is 68, measured in the number of characters of the comment. The proportion of comments in the dataset that have over 60 words is 32%. The proportion of comments that have over 100 words is 18%. And the proportion of comments that have over 200 words is only 6%. </p>

For the purpose of training model, a re-sampled dataset was applied for this project. The dataset contains 28929 records, with half classified as `non-toxic` and the other half classified as `toxic`. The dataset is split into 70: 20: 10 for training, validation and testing.


## Models and Performance

### Baseline Model: BOW

needs to fill up

### CNN Model

#### V0: 

**MODEL**

The first model has 4 different filter sizes: 2, 3, 4 and 5, each focuses on different sizes of N-grams. We assigned 64 filters of each sizes for this model. The fully-connected neural network has 256 input features and binary outputs. The model's dropout rate is set to be 0.5 to avoid over fitting. </p>

The model was trained on MacBook Pro M2 CPU for 10 epochs, which took around 30 minutes.

```python

(CNN_NLP(
   (conv1d_list): ModuleList(
     (0): Conv1d(300, 64, kernel_size=(2,), stride=(1,))
     (1): Conv1d(300, 64, kernel_size=(3,), stride=(1,))
     (2): Conv1d(300, 64, kernel_size=(4,), stride=(1,))
     (3): Conv1d(300, 64, kernel_size=(5,), stride=(1,))
   )
   (fc): Linear(in_features=256, out_features=2, bias=True)
   (dropout): Dropout(p=0.5, inplace=False)
 )
```

**PARAMETERS**

|Description         |Values           |
|:------------------:|:---------------:|
|input word vectors  |GloVe            |
|embedding size      |300              |
|filter sizes        |(2, 3, 4, 5)     |
|num filters         |(64, 64, 64, 64) |
|activation          |ReLU             |
|pooling             |1-max pooling    |
|dropout rate        |0.5              |

**PERFORMANCE** </p>

For training and validation, we see the following performance: 

| Epoch  |  Train Loss  |  Val Loss  |  Val Acc  |  Elapsed 
|:------:|:------------:|:----------:|:---------:|:-------:|
|   1    |   0.515071   |  0.387810  |   85.50   |  222.95 | 
|   2    |   0.357476   |  0.314949  |   87.84   |  214.95 |
|   3    |   0.307458   |  0.286911  |   88.58   |  217.30 |
|   4    |   0.283827   |  0.271694  |   89.29   |  212.41 |
|   5    |   0.269294   |  0.261957  |   89.29   |  217.79 |
|   6    |   0.257123   |  0.254316  |   89.50   |  238.22 |
|   7    |   0.247575   |  0.248300  |   89.76   |  223.09 |
|   8    |   0.239426   |  0.243885  |   89.81   |  216.53 |
|   9    |   0.233904   |  0.239794  |   90.08   |  210.70 |
|  10    |   0.227885   |  0.236559  |   90.17   |  221.15 |

For testing, the first CNN model performed well on 2865 testing data: </p>

| Accuracy |	Precision |	Recall	 |  F1       |
|:--------:|:------------:|:--------:|:---------:|
| 0.914455 |	0.920194  |	0.910714 |	0.91543  |






