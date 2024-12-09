
# Detecting Machine-Crafted Content in Digital Media

This project focuses on detecting machine-generated text in digital media using an LSTM-based classifier. The goal is to distinguish between human-written and machine-crafted content, enhancing trust and transparency in the digital ecosystem.

## Project Overview

With the rise of AI-generated content, it's becoming increasingly crucial to develop tools to identify and classify such text. This project utilizes an LSTM neural network to achieve this classification, employing embeddings to represent textual data and a fully connected layer for final predictions.

## Features

- **Embedding Layer**: Converts text into dense vector representations.
- **LSTM Layer**: Captures sequential dependencies in the text data.
- **Binary Classification**: Distinguishes between human-written and machine-crafted content.
- **Sigmoid Activation**: Outputs probabilities for the binary classification task.

## Installation

1. Clone this repository:
   ```bash
   git clone
   cd your-repo-name
   ```

2. Install required packages:
   ```bash
   pip install torch torchvision torchsummary
   ```

3. (Optional) Install Jupyter for interactive exploration:
   ```bash
   pip install notebook
   ```

## Usage

### Model Training

1. Prepare your dataset with labeled text (human vs. machine-generated).
2. Tokenize and preprocess the text to create sequences of indices.
3. Initialize the `LSTMClassifier` with appropriate parameters:
   ```python
   model = LSTMClassifier(vocab_size, embed_dim, hidden_dim, output_dim)
   ```
4. Train the model using an optimizer like Adam and a suitable loss function, such as binary cross-entropy.

### Prediction

Load your trained model and make predictions on new data:
```python
model.eval()
output = model(input_tensor)
prediction = (output > 0.5).int()
```

### Evaluation

Evaluate the model's performance on a test dataset using metrics like accuracy, precision, recall, and F1-score.

### Results

The LSTMClassifier achieved an accuracy of **85.67%** on the test dataset.

## Project Structure

- `data/`: Directory for storing dataset files.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and experiments.

## Dependencies

- Python 3.8 or higher
- PyTorch
- TorchSummary
- Jupyter (optional)

## Example Usage with TorchSummary

To view the model architecture:
```python
from torchsummary import summary
model = LSTMClassifier(vocab_size, embed_dim, hidden_dim, output_dim)
summary(model, input_size=(sequence_length,), device="cpu")
```

## Future Work

- Integrate pre-trained models like BERT for improved accuracy.
- Expand the model to handle multi-class classifications for different types of generated content.
- Develop a web application for real-time text classification.

## Contributing

Contributions are welcome! Please fork this repository, create a new branch, and submit a pull request.

## License

This project is licensed under the MIT License.
