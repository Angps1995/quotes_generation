# Generate new quotes using dataset of ~500,000 quotes using a basic model with GRU.

**Dataset is obtained from:**

Title: Proposing Contextually Relevant Quotes for Images

Authors: Shivali Goel, Rishi Madhok, Shweta Garg

In proceedings of: 40th European Conference on Information Retreival

Year: 2018

https://github.com/ShivaliGoel/Quotes-500K


**Data Preprocessing**

Save the dataset into 'data' folder.
Run the file text_to_vec.py under utils


**Training**

Set the parameters and run ./train.sh 


**Generating new quotes**

Run the file generate_quotes.py. The exported model currently can be found in the model_logs/export and you can use it to generate quotes.


**TODO**:

1. Implement Beam search 

2. convert to TF 2.0 once cudnn version is resolved.
