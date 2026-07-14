# BERT Text Classification for Insincere Question Detection

This project fine-tunes a pretrained BERT model to classify questions as "sincere" or "insincere," using the Quora Insincere Questions Classification dataset. Insincere questions are those based on false premises, meant to make a statement rather than genuinely seek an answer, or intended to be inflammatory or provocative. The notebook demonstrates the full workflow of adapting a pretrained transformer model to a binary text classification task: preparing raw text data, tokenizing it into BERT's expected input format, building a classification head on top of the pretrained encoder, fine-tuning the combined model, and evaluating its predictions on new sentences.

This is intended for people who want a practical, end to end example of transfer learning with BERT in TensorFlow, such as ML enthusiasts exploring NLP fine-tuning workflows, students learning how BERT based classifiers are built, or developers who need a working template for their own binary text classification tasks.

## Features

- Loads a pretrained BERT encoder (`bert_en_uncased_L-12_H-768_A-12`) directly from TensorFlow Hub / Kaggle Models.
- Converts raw question text into the three BERT input tensors: `input_word_ids`, `input_mask`, and `input_type_ids`, using the official BERT tokenizer from `tf-models-official`.
- Builds an efficient `tf.data.Dataset` pipeline with shuffling, batching, and prefetching for training and validation.
- Adds a lightweight classification head (dropout plus a dense sigmoid layer) on top of BERT's pooled output.
- Fine-tunes the full model end to end with the Adam optimizer and binary cross entropy loss.
- Plots training and validation loss/accuracy curves after training.
- Runs inference on custom example sentences and labels each as "Sincere" or "Insincere."

## Project Structure

The project is a single Jupyter notebook (`BERT_Text_Classification.ipynb`) organized into sequential tasks:

- **Task 1: Setup and imports.** Installs `tf_keras`, `tensorflow`, `tensorflow-text`, and `tf-models-official`, then imports `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `tensorflow`, `tensorflow_hub`, and BERT specific utilities from `official.nlp`.
- **Task 2: Dataset download.** Loads the Quora Insincere Questions Classification dataset directly from a compressed CSV hosted on archive.org into a pandas DataFrame, and plots the class distribution of the `target` column.
- **Task 3: Dataset splitting.** Uses `train_test_split` from scikit-learn to create small, stratified training and validation subsets, then wraps them as `tf.data.Dataset` objects.
- **Task 4: Pretrained BERT download.** Loads the BERT encoder layer and its matching tokenizer (vocabulary file and casing settings) from TensorFlow Hub.
- **Task 5 and Task 6: Tokenization.** Defines `to_feature`, which converts a single text and label pair into BERT input features using `classifier_data_lib`, and wraps it with `tf.py_function` in `to_feature_map` so it can run inside a `tf.data` pipeline in eager mode.
- **Task 7: Input pipeline.** Applies `to_feature_map` to the training and validation datasets, then shuffles, batches (batch size 32), and prefetches them for efficient training.
- **Task 8 and Task 9: Model building and fine-tuning.** Defines `create_model()`, which attaches a dropout layer and a sigmoid output layer to the BERT encoder's pooled output, compiles the model, and trains it for 4 epochs.
- **Task 10: Evaluation.** Plots loss and accuracy curves for training and validation, and runs the fine-tuned model on a handful of example sentences to classify each as sincere or insincere.

## Requirements

- Python 3 environment with internet access (to download the dataset and the pretrained BERT weights).
- **A GPU is required to run this project in a reasonable amount of time.** Fine-tuning a full BERT encoder on CPU is extremely slow and is not a practical option for this notebook.
- If running in Google Colab, select a GPU runtime before executing any cells (`Runtime > Change runtime type > Hardware accelerator > GPU`).
- If running locally or on another platform, ensure that a CUDA capable GPU, a compatible NVIDIA driver, and the correct CUDA/cuDNN versions for your TensorFlow version are installed.
- The notebook itself verifies GPU availability at runtime with:

```python
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")
```

Confirm this prints "available" before proceeding, otherwise training will be impractically slow.

## Installation

Install the required packages by running the setup cells at the top of the notebook:

```bash
pip install -q tf_keras
pip install -q -U tensorflow==2.20.* tensorflow-text==2.20.* tf-models-official==2.20.*
```

An environment variable must also be set before importing TensorFlow, so that Keras uses its legacy (`tf_keras`) implementation, which is required for compatibility with `tensorflow-text` and `tf-models-official`:

```python
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
```

## Usage

Run the notebook cells in order from top to bottom. The general flow is:

1. **Load the dataset:**

```python
df = pd.read_csv(
    'https://archive.org/download/fine-tune-bert-tensorflow-train.csv/train.csv.zip',
    compression='zip', low_memory=False
)
```

2. **Create small, stratified train/validation splits:**

```python
train_df, remaining = train_test_split(df, random_state=42, train_size=0.0075, stratify=df.target.values)
valid_df, _ = train_test_split(remaining, random_state=42, train_size=0.00075, stratify=remaining.target.values)
```

3. **Load the BERT layer and tokenizer:**

```python
bert_layer = hub.KerasLayer(
    "https://kaggle.com/models/tensorflow/bert/TensorFlow2/en-uncased-l-12-h-768-a-12/2",
    trainable=True
)
```

4. **Build and train the model:**

```python
model = create_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.BinaryAccuracy()]
)

history = model.fit(train_data, validation_data=valid_data, epochs=4, verbose=1)
```

5. **Run predictions on new text:**

```python
sample_example = [
    "I am having such a good day",
    "Kill yourself",
    "I hate people",
    "Did you guys see that new movie?"
]

test_data = tf.data.Dataset.from_tensor_slices((sample_example, [0]*len(sample_example)))
test_data = test_data.map(to_feature_map).batch(1)
preds = model.predict(test_data)

['Insincere' if pred >= 0.5 else 'Sincere' for pred in preds]
```

## Configuration

The notebook exposes a few parameters directly in code rather than through a separate config file or environment variables:

| Parameter | Location | Value | Purpose |
|---|---|---|---|
| `label_list` | Task 4 | `[0, 1]` | Binary label categories |
| `max_seq_length` | Task 4 | `128` | Maximum token length per input sequence |
| `train_batch_size` | Task 4 | `32` | Batch size used when building the input pipeline |
| Dropout rate | Task 8 (`create_model`) | `0.4` | Dropout applied to BERT's pooled output before classification |
| Learning rate | Task 9 | `2e-5` | Adam optimizer learning rate for fine-tuning |
| `epochs` | Task 9 | `4` | Number of fine-tuning epochs |
| Classification threshold | Task 10 | `0.5` | Cutoff for labeling a prediction as "Insincere" |

Adjust these values directly in the corresponding cells to change model behavior, such as raising `max_seq_length` for longer questions or increasing `epochs` for more thorough training (at the cost of the overfitting already observed at 4 epochs).

## Technologies Used

- **Language:** Python 3
- **Deep learning framework:** TensorFlow 2.20, with the `tf_keras` legacy Keras backend
- **NLP components:** TensorFlow Hub (pretrained BERT), TensorFlow Text, `tf-models-official` (BERT tokenizer and optimization utilities)
- **Data handling:** pandas, NumPy, scikit-learn (`train_test_split`)
- **Visualization:** Matplotlib
- **Dataset:** Quora Insincere Questions Classification dataset

## Author

*M. E. Petra Bošković*
