# Reuters-21578 documents topics classicication model

## Description

Reuters 21578 documents topics classicication model is a machine learning model 
which is developed for classifying the topics of a given document from Reuters-21578 
dataset as earn or not-earn classes.

## Requirements

The development process for the classification model passed through the following three steps:

step-1: Parsing the document

- The dataset, in SGM(Standard Generalized Markup Language) file format is parsed and necessary data's has been extracted.
- The data further cleaned for removal of some special characters and extra spaces.
- Generated a pandas dataframe with features of each document and the topics as a target variable.

step-2: Generate text embedding

- Apply document to sentence tokenizer (NLTK).
- Generate the embedding for the documnet using sentence embedding model (Used DistilBERT).
- Generate final embedding of a given text document.
- Binarize the topics class (1 represents topic earn and 0 represents other topics)

step-3: Train Neural Network model

- Train a Single hidden layer feed-forward Neural Network.

## How to use
Notebook files to run:

Note: these notebooks are assumed to run on google colab only.

1. Reuters_train.ipynb

Running this notebook does the following:
- Read the Reuters-21578  dataset, parse it, clean it, generate embedding, train a Neural Network model (and test model performance) 
  and generate and save the final keras model. 

The final keras model has been saved in this directory as keras_model.hd5 

If there is a need to generate the model again, do this:
    - Open this notebook
    - Make sure the runtime is set to GPU
    - Choose Restart and Run all option which will execute everything. 
Note: Expected excution time of this process with GPU will be 30 min.

Requirements:
The notebook access various files from the current google drive, please don't delete files from this drive. If you do, please replace the crosponding ID accordingly. 

2. Reuters_test.ipynb

Run this notebook to do test on the model generated. 

Requirements:
- The test files should be in SGML file format.
- It is expected for the topics for each document to be parsed in the SGM file. If the given test data files don't include the topics and you want to provide it in a separate file, pass the topics list as **`y_true`** variable in the do_prediction function. 

If the topics are not found from the document and the **`y_true`** variable  is set to false, then the topics are assumed to be of 0 class (not-earn).  

steps:
    - Open this notebook
    - Make sure the runtime is set to GPU
    - Upload test_data to the current google colab session
    - Add the following:
      - Set test_data_path to the test directory uploaded
        Note: By default test_data_path is set to "/content" and need to be updated accordingly.
        For example: 
        - If the name of the uploaded directory (containing the test_data's) is Reuters_21578_test, then
            test_data_path will be set to "/content/Reuters_21578_test"
      - Set y_true variable to the do_prediction function, if there is a need to peovide topics for documents of the test_data in a separate file 
    - Choose Restart and Run all option and execute everything.

## References
1. http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/
2. https://skimai.com/fine-tuning-bert-for-sentiment-analysis/
3. http://jalammar.github.io/illustrated-bert/
4. https://colab.research.google.com/github/jalammar/jalammar.github.io/blob/master/notebooks/bert/A_Visual_Notebook_to_Using_BERT_for_the_First_Time.ipynb#scrollTo=FoCep_WVuB3v
5. https://www.analyticsvidhya.com/blog/2021/05/a-complete-hands-on-guide-to-train-your-neural-network-model-on-google-colab-gpu/


