## Sarcasm detection with LSTM neural networks and GloVe embedding

This application detects sarcasm in sentences. For this, two different LSTM neural network models were trained, one with the default tensorflow embedding trained together with the neural network and the other using GloVe for embedding.

Both reached an accuracy of 85%.

First it is necessary to install the streamlit library with the following command:

`pip install streamlit`

To run the application we will write the following command in the terminal:

`streamlit run sarcasm_detection_LSTM_web.py`

Here the file name is 'sarcasm_detection_LSTM_web.py'

With this, it will show us the url that we can enter in the browser to open the application:

`Local URL: http: // localhost: 8501`
`Network URL: http: //192.168.1.106: 8501`