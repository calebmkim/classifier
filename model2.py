
from simpletransformers.classification import ClassificationModel
from bs4 import BeautifulSoup


from sentence_transformers import SentenceTransformer
from joblib import dump, load

class_list = ['The Classifieds', 'Entertainment!', 'How to DiS',
       'Music', 'News & Politics', 'Site Feedback', 'Social', 'Sports']

def clean_text(text):
    """
        text: a string
        return: modified initial string
    """
    text = BeautifulSoup(text, "lxml").text # HTML decoding
    text = text.lower() # lowercase text
    #text = ' '.join(word for word in text.split() if word not in stop_words)
    #text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    return text


# define hyperparameter
train_args ={"reprocess_input_data": True,
             "overwrite_output_dir": True,
             "fp16":False,
             "num_train_epochs": 4}

# Create a ClassificationModel
model = ClassificationModel(
    "bert", "outputs_bert_uncased/",
    num_labels=8,
    use_cuda = False
)

def predict(post):
    post[0]= clean_text(post[0])
    predictions, raw_outputs = model.predict(post)
    return class_list[predictions[0]]


'''
clf = load('classifier.joblib')

transformer = SentenceTransformer("paraphrase-mpnet-base-v2")

def predict(post):
    post_embedding = transformer.encode(post)
    prediction = clf.predict(post_embedding)
    return prediction[0]
    '''
