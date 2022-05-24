from bs4 import BeautifulSoup
import os
import numpy as np
import pandas as pd
from string import punctuation
import re
import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import nltk
import tensorflow as tf

class reuters_util:
    model = ""
    tokenizer = ""
    # Parses a document in sgm (Standard Generalized Markup language)
    # return a pandas dataframe with desired features. 
    def parse_doc(self, doc, columns, df):
        try:
            current_doc = BeautifulSoup(open(doc), features='lxml')
        except UnicodeDecodeError:
            current_doc = BeautifulSoup(open(doc, 'rb'), features='lxml', from_encoding="iso-8859-1")
        doc_list = current_doc.find_all('reuters')
        for d in doc_list:
            row=[]
            for col in columns:
                try:
                    if col in ["places","topics", "orgs", "people","exchanges"]:
                        result = [i.text for i in d.find(col).contents]
                        if result:
                            row.append(result)
                        else:
                            row.append(None)
                    elif col in ['cgisplit', 'lewissplit', 'newid', 'topics_bool']:
                        if col == "topics_bool":
                            row.append(d.get("topics"))
                        else:
                            row.append(d.get(col))
                    else:
                        row.append(str(d.find(col).contents[-1]).strip())
                except :
                    row.append(None)
            df.loc[df.shape[0]]= row
        
        return df

    # clean the data
    def clean_data(self, df):
        special_char = list(punctuation)
        for e in ['.','?']:
            special_char.remove(e)
        special_char.append("\n+")
        special_char.append("\s+")

        def deep_clean(text_str):
            text_str = str(text_str)
            text_str = re.sub('<[^>]*>', '', text_str)
            for char in special_char:
                text_str = text_str.replace(char, '')
            return text_str

        df['text'] = df['text'].apply(deep_clean)
        df['title'] = df['title'].apply(deep_clean)
        return df

    # Configure GPU
    def gpu_config(self):
        # Get the GPU device name.
        device_name = tf.test.gpu_device_name()

        # The device name should look like the following:
        if device_name == '/device:GPU:0':
            print('Found GPU at: {}'.format(device_name))
        else:
            raise SystemError('GPU device not found')

        # If there's a GPU available...
        if torch.cuda.is_available():    
            # Tell PyTorch to use the GPU.    
            device = torch.device("cuda")
            print('There are %d GPU(s) available.' % torch.cuda.device_count())
            print('We will use the GPU:', torch.cuda.get_device_name(0))

        # If not...
        else:
            print('No GPU available, using the CPU instead.')
            device = torch.device("cpu")

    def initialize_prebuilt_models(self):
        # DistilBERT model
        model_class, pretrained_weights = (transformers.DistilBertModel, 'distilbert-base-uncased')

        # Load pretrained model
        self.model = model_class.from_pretrained(pretrained_weights)

        # Use GPU
        self.model.cuda()

        # Load the BERT tokenizer
        self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        # Download NLTK for paragraph tokenization
        nltk.download('punkt')

    def doc_to_vectors(self, doc):
        sentences  = nltk.sent_tokenize(doc)

        def valid_sent(sent):
            return True if len(sent.split(" ")) > 1 else False

        vector = np.array([self.sent_to_vector(s) for s in sentences if valid_sent(s)]).sum(axis=0)
        return vector

    def sent_to_vector(self, sent):
        # Encode the sentence
        encoded = self.tokenizer.encode_plus(
            text=sent,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length = 64,  # maximum length of a sentence
            pad_to_max_length=True,  # Add [PAD]s
            truncation = True, # Truncate words beyond the max_length
            return_attention_mask = True,  # Generate the attention mask
            return_tensors = 'pt',  # ask the function to return PyTorch tensors
        )

        # Get the input IDs and attention mask in tensor format
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']

        with torch.no_grad():
            last_hidden_states = self.model(input_ids.to("cuda"), attention_mask=attention_mask.to("cuda"))
        #BERT adds a token called [CLS] (for classification) at the beginning of every sentence, we take that
        features = last_hidden_states[0][:,0,:].cpu().numpy()
        return features


    def generate_final_embedding(self, df):
        result_data = []

        def padding_na(vector):
            std_vector_length = 768
            if not vector.shape:
                return np.array([0.0]*std_vector_length).reshape(1,std_vector_length)
            else:
                return vector

        for i in range(len(df)):
            try:
                txt_emb = df.iloc[i]['text_embedding']
                title_emb = df.iloc[i]['title_embedding']
                result_data.append(np.concatenate((txt_emb, title_emb), axis=1)[0])
            except ValueError:
                result_data.append(np.concatenate((padding_na(txt_emb), padding_na(title_emb)), axis=1)[0])

        result_df = pd.DataFrame(result_data)
        result_df['topics'] = df['topics']
        return result_df

    # Test data preprocessing function
    def preprocess_data(self, datapath):
        columns = ['title','text','topics']
        test_df = pd.DataFrame(columns=columns)
        test_files = [f for f in os.listdir(datapath) if f.endswith("sgm")]
        if test_files:
            for test_file in test_files:
                print("parsing {}".format(test_file))
                file_path = os.path.join(datapath, test_file)
                test_df = self.parse_doc(file_path, columns, test_df)
            
                test_df = self.clean_data(test_df.copy())
                test_df['text_embedding'] = test_df['text'].apply(self.doc_to_vectors)
                test_df['title_embedding'] = test_df['title'].apply(self.doc_to_vectors)
                test_df.loc[test_df['topics'].isnull(),'topics']=['0']
                test_df['topics'] = [1 if 'earn' in topic else 0 for topic in test_df['topics']]
                test_df = self.generate_final_embedding(test_df) 
        else:
            raise ValueError('No .sgm file in a given path.')
        return test_df
