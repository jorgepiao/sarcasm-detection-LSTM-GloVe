
import streamlit as st
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model


# Cargar los modelos
model_LSTM = load_model('modelo/sarcasm_LSTM_model.h5')
model_GloVe = load_model('modelo/sarcasm_GloVe_model.h5')


def stopwords_filter(sentence):
    stopwords = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]

    headline = []
    filtered_list = []
    for word in sentence.split():
        if word not in stopwords:
            filtered_list.append(word)
    join_str = ' '.join([str(ele) for ele in filtered_list])
    headline.append(join_str)

    return headline


def tokenize(headines_txt, headline):
    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(headines_txt)

    sequences = tokenizer.texts_to_sequences(headline)
    pad_sequ = pad_sequences(sequences, maxlen=32, padding='post', truncating='post')

    return pad_sequ


with open('dataset/headlines.txt','r', encoding='utf-8') as sentences: 
    headines_txt = [line.strip() for line in sentences]




def main():
    # Titulo general
    st.title('Sarcasm Detection')
    # Titulo de la barra lateral
    st.sidebar.header('Enter Parameters')

    # Parametros de la barra lateral
    sentence = st.sidebar.text_input("Sentence")
    model = ["LSTM", "GloVe"]
    selec_model = st.sidebar.selectbox("Choose the Model", model)

    # Boton para generar el texto con los parametros elegidos
    if st.button('Check'):
        if selec_model == 'LSTM':
            model = model_LSTM

        elif selec_model == 'GloVe':
            model = model_GloVe

        headline = stopwords_filter(sentence)

        pad_sequ = tokenize(headines_txt, headline)

        prediction = (model.predict(pad_sequ) > 0.5).astype("int32")

        if prediction[0]==1:
            prediction = 'Sarcastic'
        else:
            prediction = 'NOT sarcastic'

        st.write(sentence)
        st.success(prediction)



if __name__ == '__main__':
    main()

