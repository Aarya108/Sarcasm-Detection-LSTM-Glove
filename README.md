# Detection of Sarcastic Sentences from Social Media Dataset

This project focuses on detecting sarcasm in social media headlines using deep learning techniques. By leveraging pre-trained GloVe embeddings and an LSTM-based model, this approach automatically classifies headlines as either sarcastic or non-sarcastic. The complete process—from data preprocessing and embedding to training and evaluating the model—is detailed below, along with visualization outputs and instructions for running the project on your computer.

---

## 1. Topic & Model Architecture

**Topic:**  
Detection of sarcastic sentences from a social media dataset.

**Model Architecture:**  
- **Embedding Layer:**  
  Uses pre-trained GloVe embeddings (Twitter, 100-dimension) to convert words into numerical vectors. The embedding weights remain static (non-trainable) to preserve pre-learned semantic relationships.
- **LSTM Layer:**  
  A Long Short-Term Memory (LSTM) layer with 64 units is used to capture contextual dependencies in the text with dropout and recurrent dropout applied to reduce overfitting.
- **Output Layer:**  
  A Dense layer with a sigmoid activation function classifies each headline as sarcastic (1) or non-sarcastic (0).
  

---

## 2. Preprocessing

**Steps Involved:**
- **Data Loading & Merging:**  
  Two JSON datasets containing social media headlines are loaded and merged.
- **Cleaning the Data:**  
  - Converting text to lowercase.
  - Removing URLs, punctuation, and emojis.
  - Expanding common contractions.
- **Tokenization & Stopword Removal:**  
  - Headlines are tokenized using NLTK.
  - Punctuation and non-alphabetic tokens are removed.
  - Stopwords are filtered out to retain only the most meaningful words.
- **Padding:**  
  Converted sequences are padded to a fixed length of 25 words to ensure a uniform input size for the model.
- **Model Architecture**  
![Model architecture](https://github.com/Aarya108/Sarcasm-Detection-LSTM-Glove/blob/main/results/model_architecture.png)
---

## 3. Embedding

**Pre-trained Embeddings:**
- **GloVe Embeddings:**  
  A 100-dimensional pre-trained GloVe embeddings file (e.g., `glove.twitter.27B.100d.txt`) is used.
- **Embedding Matrix:**  
  An embedding matrix is created by mapping each word from the tokenizer’s vocabulary to its corresponding GloVe vector.
- **Embedding Layer Implementation:**  
  The pre-trained embedding matrix is loaded into the embedding layer of the model, which is set as non-trainable to preserve the semantic information.

---

## 4. Model Training

**Architecture Recap:**
- **Input:**  
  Padded sequences of length 25.
- **Embedding Layer:**  
  Incorporates the pre-trained GloVe matrix.
- **LSTM Layer:**  
  64 LSTM units with dropout regularization.
- **Output Layer:**  
  Sigmoid activated dense layer for binary classification.

**Training Details:**
- **Compilation:**  
  The model is compiled using the Adam optimizer and binary cross-entropy loss.
- **Training:**  
  Trained for 25 epochs with a batch size of 32 and a validation split of 20%.
- **Performance Metrics:**  
  - Accuracy: ~91%
  - Precision: ~91.3%
  - Recall: ~88.8%
  - F1 Score: ~90.0%

---

## 5. Results Visualization

Several visualizations have been generated to evaluate model performance and compare it with other approaches. The uploaded images are integrated below:

### Data Distribution and Word Cloud
- **Word Cloud (Overall):**  
  This image shows the distribution of the most common words from the dataset.
  
  ![Word Cloud](https://github.com/Aarya108/Sarcasm-Detection-LSTM-Glove/blob/main/results/wordcloud.png)

### Model Performance Comparison
- **Model Comparison:**  
  This image compares the performance of the sarcasm detection model against other models tested.
  
  ![Model Comparison](https://github.com/Aarya108/Sarcasm-Detection-LSTM-Glove/blob/main/results/sarcasm_model_comparison.png)

### Prediction Distribution
- **Prediction Probability Distribution:**  
  A histogram displaying the distribution of predicted probabilities.
  
  ![Prediction Distribution](https://github.com/Aarya108/Sarcasm-Detection-LSTM-Glove/blob/main/results/predprob.png)

### Evaluation Metrics
- **ROC Curve:**  
  Depicts the Receiver Operating Characteristic curve of the model.
  
  ![ROC Curve](https://github.com/Aarya108/Sarcasm-Detection-LSTM-Glove/blob/main/results/rocsarcsam.png)
  
- **Precision-Recall Curve:**  
  Shows the trade-off between precision and recall.
  
  ![Precision-Recall Curve](https://github.com/Aarya108/Sarcasm-Detection-LSTM-Glove/blob/main/results/prsarcasm.png)
  
- **Confusion Matrix:**  
  A visualization of true and false positives and negatives, compared with other approaches.
  
  ![Confusion Matrix Comparison](https://github.com/Aarya108/Sarcasm-Detection-LSTM-Glove/blob/main/results/cfsarcsasm.png)

---

## 6. How to Run on Your PC

**Requirements:**
- **Python 3.x**  
- **Libraries:** TensorFlow, Keras, numpy, pandas, matplotlib, nltk, seaborn, scikit-learn, and wordcloud.

**Installation Steps:**
1. **Clone or Download the Repository.**
2. **Install Dependencies:**  
   Run the following command in your terminal:
   ```bash
   pip install tensorflow numpy pandas matplotlib nltk seaborn scikit-learn wordcloud
   ```
3. **Dataset & Embedding Files:**  
   - Download the JSON datasets (e.g., `Sarcasm_Headlines_Dataset.json` and `Sarcasm_Headlines_Dataset_v2.json`).
   - Download the pre-trained GloVe embeddings file (e.g., `glove.twitter.27B.100d.txt`) and place it into the designated directory.
4. **Run the Notebook:**
   - Open `Sarcasm_Colab_Ready.ipynb` in Jupyter Notebook or Google Colab.
   - Execute the cells sequentially to preprocess the data, build, train, evaluate, and save the model.
5. **Inference:**
   - Utilize the provided function `predict_sarcasm(s)` to classify new headlines.
   - The trained model is saved as `trained_model.h5` and can be reloaded for further testing or API deployment.
6. **API Deployment (Optional):**
   - For deploying the model using an API framework, refer to the code snippet provided in the notebook for converting the model into an API-ready format.

---

