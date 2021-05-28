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
                           Layer (type)                 Output Shape              Param #   

                           =================================================================
                           input_10 (InputLayer)        [(None, 50)]              0         
                           _________________________________________________________________
                           embedding_9 (Embedding)      (None, 50, 100)           1141000   
                           _________________________________________________________________
                           spatial_dropout1d_5 (Spatial (None, 50, 100)           0         
                           _________________________________________________________________
                           lstm_1 (LSTM)                (None, 100)               80400     
                           _________________________________________________________________
                           dense_26 (Dense)             (None, 50)                5050      
                           _________________________________________________________________
                           dropout_5 (Dropout)          (None, 50)                0         
                           _________________________________________________________________
                           dense_27 (Dense)             (None, 11)                561       
                           =================================================================
                           Total params: 1,227,011
                           Trainable params: 86,011
                           Non-trainable params: 1,141,000
                           _________________________________________________________________
9. **Gated Recurrent Units(GRU)**
                         Layer (type)                 Output Shape              Param #   

                        =================================================================
                        input_9 (InputLayer)         [(None, 50)]              0         
                        _________________________________________________________________
                        embedding_8 (Embedding)      (None, 50, 100)           1141000   
                        _________________________________________________________________
                        spatial_dropout1d_4 (Spatial (None, 50, 100)           0         
                        _________________________________________________________________
                        gru_2 (GRU)                  (None, 100)               60600     
                        _________________________________________________________________
                        dense_24 (Dense)             (None, 50)                5050      
                        _________________________________________________________________
                        dropout_4 (Dropout)          (None, 50)                0         
                        _________________________________________________________________
                        dense_25 (Dense)             (None, 11)                561       
                        =================================================================
                        Total params: 1,207,211
                        Trainable params: 66,211
                        Non-trainable params: 1,141,000
                        _________________________________________________________________
### 5 Comparing Various models
                                                Model       Feature  accuracy    recall  precision  f1_score

                              0   LogisticRegression  count_vector  0.665052  0.665052   0.677301  0.654163
                              1   LogisticRegression     word_tfid  0.653821  0.653821   0.700271  0.630635
                              2   LogisticRegression    ngram_tfid  0.471482  0.471482   0.698646  0.379844
                              3   LogisticRegression     char_tfid  0.629597  0.629597   0.656482  0.603447
                              4   LogisticRegression        word2v  0.393746  0.393746   0.319688  0.240149
                              5   LogisticRegression         doc2v  0.636203  0.636203   0.629743  0.621049
                              6           NaiveBayes  count_vector  0.662850  0.662850   0.705685  0.634743
                              7           NaiveBayes     word_tfid  0.522132  0.522132   0.684921  0.436509
                              8           NaiveBayes    ngram_tfid  0.439991  0.439991   0.623122  0.324472
                              9           NaiveBayes     char_tfid  0.509139  0.509139   0.651471  0.422890
                              10                 SVM  count_vector  0.664391  0.664391   0.660724  0.657461
                              11                 SVM     word_tfid  0.668355  0.668355   0.673203  0.649724
                              12                 SVM    ngram_tfid  0.596124  0.596124   0.660077  0.569930
                              13                 SVM     char_tfid  0.630918  0.630918   0.627345  0.605745
                              14                 SVM        word2v  0.383396  0.383396   0.446595  0.287407
                              15                 SVM         doc2v  0.581810  0.581810   0.574610  0.570994
                              16        RandomForest  count_vector  0.626074  0.626074   0.637684  0.608010
                              17        RandomForest     word_tfid  0.643250  0.643250   0.676724  0.619075
                              18        RandomForest    ngram_tfid  0.421053  0.421053   0.623688  0.455096
                              19        RandomForest     char_tfid  0.593041  0.593041   0.671430  0.543571
                              20        RandomForest        word2v  0.561771  0.561771   0.611908  0.522000
                              21        RandomForest         doc2v  0.598547  0.598547   0.640228  0.540541
                              22                 XGB  count_vector  0.549218  0.549218   0.643464  0.509332
                              23                 XGB     word_tfid  0.551641  0.551641   0.657085  0.509858
                              24                 XGB    ngram_tfid  0.448800  0.448800   0.695761  0.347549
                              25                 XGB     char_tfid  0.576965  0.576965   0.667935  0.530469
                              26                 XGB        word2v  0.526536  0.526536   0.543025  0.477679
                              27                 XGB         doc2v  0.616164  0.616164   0.624451  0.582086
                              28                  NN  count_vector  0.688174  0.688174   0.692654  0.610068
                              29                  NN     word_tfid  0.699405  0.699405   0.697822  0.611736
                              30                  NN        word2v  0.393966  0.393966   0.331468  0.077667
                              31                  NN         doc2v  0.640828  0.640828   0.635525  0.557621
                              32                 CNN      word2vec  0.481832  0.481832   0.466516  0.234011
                              33                CNN2      word2vec  0.408060  0.408280   0.391929  0.271799
                              34                LSTM      word2vec  0.486677  0.486677   0.480852  0.274471
                              35                 GRU      word2vec  0.492182  0.492182   0.464086  0.296559
                              
   Here we actually can compare how good our model performed on test sets. Accuracies are quite low in the range of 50-60% because the text provided was very short and to          extract particular pattern from it is difficult. Developing logistic regression , SVM and shallow neural networks on count vectors and word_tfid seems promising but it can      further be developed by tuning hyper parameters. Deep Neural networks could even perform better if large amount of data would be provided to train a embedding matrix. Also      this models can be furthered tuned and trained in larger epochs keeping in mind overfitting of the data.
   
   some ways of further developing models can be-
   **Text Cleaning**- removing noices could certainly improve accuracies.
   **Tuning Hyper parameter** cross validation set can be used to tune the parameters
   **Stacking Models** - Different models can be stacked together to make a whole single model to perform better.
   **Feature Learning**- two or more features could be combined to make a single feature. That may improve our model.
   

