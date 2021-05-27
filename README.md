# Text_classification

Datasheet consist of two meta data:
1. Text 
2. Label 

There are 11 Labels named **['Bigdata', 'Blockchain', 'Cyber Security', 'Data Security', 'FinTech', 'Microservices', 'Neobanks', 'Reg Tech', 'Robo Advising', 'Stock Trading', 'credit reporting' ]**
Each text data in each row is some description about its following label.

Our task is to train a multi class text classification model that predict label of the given text.

Approach to classify data consists of following steps:
### 1.  Data Reading -
given data sheet is read in pandas dataframe 
### 2. Data preprocessing -
   1. Raw data provided consists some missing data or abrubt value so that is handled using pandas libraries
   2. Text Needs to be cleaned. Text may consist of non word characters, symbols extra whitespaces. Also Text needs to be converted single case that is either lowercase or               uppercase 
   3. Removing stopwords - Stopwords are the English words which does not add much meaning to a sentence. They can safely be ignored without sacrificing the meaning of the               sentence. For example, the words like the, he, have etc.
   4. Encoding Categorial data - Our labels consist of string which needs to be encoded. we can either use Label encoder that specifies number to each label or One Hot encoding         that creates matrix of binary values. In this classification we will need both type of encoding some models will be trained with label encoding and some with one hot               encoding.
   5. Splitting the data into Train and Test set. Train set will be used to train and fit the model and test set will be used to evaluate the model.
### 3. Feature Extraction-
Cleaned text is transformed into meaningful feature vectors that are indeed used to fit the model. raw text cannot be directly fed into machine as it is strings and model cannot be trained on raw strings. There are several techniques of Featuring the text we will try them on models and select one with better results.
Some features that I have tried are-
1. Count Vectors
2. Tfid 
   -word
   -ngram 
   -character
3. Word2vec
4. Doc2vec
5. Glove using pretrained embedding model.[glove.6B.300d.txt](http://nlp.stanford.edu/data/glove.6B.zip)

### 4. Model Building - 
There are various machine and deep learning classifiction models that can be used to fit the training set. I trained model for the each of the feature and and then finally compare the evaluation metrics.
Machine Learning models that are trained are - 
1. **Logistic Regression** - This is linear model that fits the data and classify the test data
2. **Naive Bayes**- Multinomial Naïve Bayes consider probabilty of a feature vector where a given term represents the number of times it appears or very often i.e. frequency
3. **Support Vector Machine** -  SVM algorithm creates the best line or decision boundary that segregate our 11-dimensional space into classes and predict easily future classification on hyperplane
4. **Random Forest Classifier** - uses number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy
5. **Extreme Gradient Boosting(XGB)**- This model is built performing gradient boosting and training weak parameters.
6. **Neural network**-

                          Layer (type)                 Output Shape              Param #   
                          =================================================================
                          input_8 (InputLayer)         [(None, 10506)]           0         
                          _________________________________________________________________
                          dense_18 (Dense)             (None, 100)               1050700   
                          _________________________________________________________________
                          dense_19 (Dense)             (None, 11)                1111      
                          =================================================================
                          Total params: 1,051,811
                          Trainable params: 1,051,811
                          Non-trainable params: 0
                          _________________________________________________________________
  creating three layer shallow neural network. input layer consist of feature vector dimension hidden layer is dense layer with dimension of 100 and relu activation function       last layer is output sigmoid layer.

7. **Convolutional Neural Network(CNN)**-


                          Layer (type)                 Output Shape              Param #   
                          =================================================================
                          embedding (Embedding)        (None, 50, 300)          3423000   
                          _________________________________________________________________
                          conv1d (Conv1D)              (None, 174, 32)           76832     
                          _________________________________________________________________
                          max_pooling1d (MaxPooling1D) (None, 87, 32)            0         
                          _________________________________________________________________
                          flatten (Flatten)            (None, 2784)              0         
                          _________________________________________________________________
                          dense (Dense)                (None, 25)                69625     
                          _________________________________________________________________
                          dense_1 (Dense)              (None, 11)                286       
                          =================================================================
                          Total params: 3,569,743
                          Trainable params: 146,743
                          Non-trainable params: 3,423,000
                          _________________________________________________________________
  creating a embedding layer from word2vec trained earlier where each word is mapped to 300 dimensional vector. next is convolutional 1D layer of 32 filters followed by max       pooling layer of pool size =2  flattening it and creating two dense layer of 25 and number of labels(11) respectively.
  
8. **Long Short Term Memory(LSTM)** 
9. **Gated Recurrent Units(GRU)**


