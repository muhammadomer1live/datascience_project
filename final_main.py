# PyTorch modules for building and training neural networks
import torch  # Core PyTorch functionality
import torch.nn as nn  # Neural network layers and functions
import torch.optim as optim  # Optimization algorithms

# PyTorch utilities for data handling and preprocessing
from torch.utils.data import DataLoader, Subset  # DataLoader for batching and Subset for dataset manipulation

# torchvision for datasets, transformations, and pre-trained models
from torchvision import datasets, transforms, models  # Common datasets, image transformations, and model architectures

# pycocotools for handling COCO dataset annotations
from pycocotools.coco import COCO  # API to load COCO dataset annotations

# Standard libraries for numerical operations and time management
import numpy as np  # Numerical operations and array handling
import time  # Time-related functions
import os  # Operating system interactions

# PIL for image processing
from PIL import Image  # Image processing and manipulation
from PIL import ImageTk  # Tkinter support for PIL images

# Collections for specialized container data types
from collections import Counter  # Efficient counting of hashable objects

# Matplotlib for plotting and visualization
import matplotlib.pyplot as plt  # Basic plotting functions
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # Integration with Tkinter for displaying plots

# Seaborn for statistical data visualization
import seaborn as sns  # High-level interface for drawing attractive statistical graphics

# tqdm for progress bars in loops
from tqdm import tqdm  # Progress bar utility for loops

# Tkinter for GUI applications
import tkinter as tk  # Basic GUI functionalities
from tkinter import ttk, messagebox, filedialog  # Additional Tkinter widgets and dialogs

# psutil for system monitoring
import psutil  # Utilities for system and process monitoring

# rouge_score for evaluating text generation models
from rouge_score import rouge_scorer  # ROUGE score computation for evaluating generated text

# NLTK for evaluating text generation with BLEU scores
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction  # BLEU score computations for text evaluation

# JSON for saving and loading data
import json  # JSON serialization and deserialization

# Set up visualization style
sns.set(style="whitegrid")

# Global lists to store various values for metrics and losses
epoch_losses = []
val_losses = []
train_losses = epoch_losses

nic_rouge_scores = []
scst_rouge_scores = []
nic_bleu_score = []
scst_bleu_score = []
nic_caption_lengths = []
scst_caption_lengths = []
captions_nic = []
captions_scst = []

# Function to build vocabulary from COCO captions
def build_coco_vocab(coco, min_freq=5):
    """
    Builds a vocabulary from COCO captions based on a minimum word frequency.

    Args:
        coco: COCO object containing the dataset annotations.
        min_freq: Minimum frequency of words to be included in the vocabulary.

    Returns:
        word_to_idx: Dictionary mapping words to indices.
        idx_to_word: Dictionary mapping indices to words.
    """
    ann_ids = coco.getAnnIds()
    anns = coco.loadAnns(ann_ids)
    
    # Collect all captions from annotations
    captions = [ann['caption'] for ann in anns]
    tokenized_captions = [caption.lower().split() for caption in captions]
    
    # Count word frequencies
    word_counter = Counter(word for caption in tokenized_captions for word in caption)
    
    # Build vocabulary with words occurring more than 'min_freq' times
    vocab = [word for word, freq in word_counter.items() if freq >= min_freq]
    vocab = ['<PAD>', '<START>', '<END>', '<UNK>'] + vocab  # Add special tokens
    
    # Create mapping of words to indices and vice versa
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    
    return word_to_idx, idx_to_word

# Function to tokenize captions
def tokenize_caption(caption, word_to_idx, max_length=20):
    """
    Tokenizes a caption by converting it into a list of indices based on the vocabulary.

    Args:
        caption: The caption to be tokenized.
        word_to_idx: Dictionary mapping words to indices.
        max_length: Maximum length of tokenized captions.

    Returns:
        A tensor of tokenized caption of specified max_length, padded if necessary.
    """
    tokens = [word_to_idx.get(word, word_to_idx['<UNK>']) for word in caption.lower().split()]
    tokens = tokens[:max_length]  # Truncate if exceeds max_length
    tokens += [word_to_idx['<PAD>']] * (max_length - len(tokens))  # Pad to max_length
    return torch.tensor(tokens, dtype=torch.long)

# Function to detokenize captions
def detokenize_caption(tokens, idx_to_word):
    """
    Converts a sequence of tokens back into a human-readable caption, excluding special tokens.

    Args:
        tokens: Tensor of tokenized caption.
        idx_to_word: Dictionary mapping indices to words.

    Returns:
        A string representing the detokenized caption.
    """
    caption = []
    for token in tokens:
        word = idx_to_word.get(token, '')
        if word in ['<PAD>', '<START>', '<END>', '<UNK>']:  # Skip special tokens
            continue
        caption.append(word)
    return ' '.join(caption)

# Function to collate data in the DataLoader
def coco_collate_fn(batch, word_to_idx, max_length=20):
    """
    Custom collate function for handling a batch of COCO captions and images.

    Args:
        batch: A batch of data containing images and their corresponding captions.
        word_to_idx: Dictionary mapping words to indices.
        max_length: Maximum length of the tokenized captions.

    Returns:
        images: Tensor of images in the batch.
        captions: Tensor of tokenized captions in the batch.
    """
    images = []
    captions = []
    
    for image, caption in batch:
        images.append(image)
        tokenized_caption = tokenize_caption(caption[0], word_to_idx, max_length)
        captions.append(tokenized_caption)
    
    images = torch.stack(images, dim=0)  # Stack images along a new dimension
    captions = torch.stack(captions, dim=0)  # Stack captions similarly
    return images, captions

# Root directory for the COCO dataset
coco_root = '/mnt/c/Users/muham/Downloads/coco7/'

# Transformations to apply to images for data augmentation and normalization
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize all images to 128x128
    transforms.RandomHorizontalFlip(),  # Random horizontal flipping
    transforms.RandomRotation(15),  # Random rotation between -15 and 15 degrees
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet means and stds
])

# Load COCO train and validation datasets
coco_train = COCO(f"{coco_root}/annotations/captions_train2017.json")
coco_val = COCO(f"{coco_root}/annotations/captions_val2017.json")
word_to_idx, idx_to_word = build_coco_vocab(coco_train)

# Function to create a random subset of the dataset for faster experimentation
def create_subset(dataset, max_size=100):
    """
    Creates a random subset of the dataset with a specified maximum size.

    Args:
        dataset: The full dataset to subset.
        max_size: The maximum number of samples in the subset.

    Returns:
        A Subset object containing 'max_size' random samples from the dataset.
    """
    indices = np.random.choice(len(dataset), max_size, replace=False)
    indices = list(map(int, indices))  # Ensure indices are integers
    return Subset(dataset, indices)

# Function to load a dataset from COCO images and annotations
def load_dataset(root, annFile, max_size):
    """
    Loads the COCO dataset and returns a subset with specified size.

    Args:
        root: The root directory where COCO images are stored.
        annFile: The annotation file for COCO captions.
        max_size: The maximum number of samples in the dataset subset.

    Returns:
        A subset of the COCO dataset with transformations applied.
    """
    dataset = datasets.CocoCaptions(
        root=root,
        annFile=annFile,
        transform=transform  # Apply preprocessing transformations
    )
    return create_subset(dataset, max_size)

# Load train and validation datasets with subsets for faster iteration
train_dataset = load_dataset(f"{coco_root}/train2017", f"{coco_root}/annotations/captions_train2017.json", max_size=100)
val_dataset = load_dataset(f"{coco_root}/val2017", f"{coco_root}/annotations/captions_val2017.json", max_size=100)

# Function to get DataLoader for training or validation datasets
def get_dataloader(dataset, batch_size=4):
    """
    Creates a DataLoader for a given dataset.

    Args:
        dataset: The dataset for which DataLoader is needed.
        batch_size: The number of samples per batch.

    Returns:
        A DataLoader object to iterate over the dataset.
    """
    return DataLoader(dataset, batch_size=batch_size, collate_fn=lambda x: coco_collate_fn(x, word_to_idx))

# Get DataLoader objects for training and validation datasets
train_dataloader = get_dataloader(train_dataset)
val_dataloader = get_dataloader(val_dataset)


class ImageCaptioningModel(nn.Module):
    def __init__(self, cnn_model, vocab_size, embedding_dim=256, hidden_dim=256, num_layers=1):
        """
        Initializes the ImageCaptioningModel that uses a CNN for image feature extraction and an LSTM for caption generation.

        Args:
            cnn_model: Pretrained CNN model for image feature extraction.
            vocab_size: Size of the vocabulary for the captions.
            embedding_dim: Dimension of the embedding space for word tokens (default=256).
            hidden_dim: Hidden state size of the LSTM (default=256).
            num_layers: Number of LSTM layers (default=1).
        """
        super(ImageCaptioningModel, self).__init__()
        
        # CNN for image feature extraction
        self.cnn = cnn_model
        
        # Embedding layer for the captions
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM for generating sequences
        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        
        # Fully connected layers for combining image features with caption embeddings and generating final output
        self.fc_input = nn.Linear(512 + embedding_dim, embedding_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, images, captions):
        """
        Forward pass for generating captions from input images.

        Args:
            images: Tensor of images of shape (batch_size, channels, height, width).
            captions: Tensor of tokenized captions of shape (batch_size, seq_length).

        Returns:
            outputs: Tensor of shape (batch_size, seq_length, vocab_size) containing predicted word scores.
        """
        # Extract image features from the CNN
        features = self.cnn(images).unsqueeze(1)  # Shape: (batch_size, 1, 512)

        # Embed the input captions
        captions_embed = self.embedding(captions)  # Shape: (batch_size, seq_length, embedding_dim)
        
        # Repeat image features for each caption word
        features = features.repeat(1, captions_embed.size(1), 1)  # Shape: (batch_size, seq_length, 512)
        
        # Concatenate image features with caption embeddings
        inputs = torch.cat((features, captions_embed), dim=2)  # Shape: (batch_size, seq_length, 512 + embedding_dim)
        inputs = self.fc_input(inputs)  # Shape: (batch_size, seq_length, embedding_dim)
        
        # Pass concatenated inputs through LSTM
        outputs, _ = self.rnn(inputs)
        
        # Generate final word predictions
        outputs = self.fc(outputs)  # Shape: (batch_size, seq_length, vocab_size)
        
        return outputs

class SCSTModel(nn.Module):
    def __init__(self, cnn_model, vocab_size, embedding_dim=256, hidden_dim=256, num_layers=1):
        """
        Initializes the Self-Critical Sequence Training (SCST) model.

        Args:
            cnn_model: Pretrained CNN model for image feature extraction.
            vocab_size: Size of the vocabulary for the captions.
            embedding_dim: Dimension of the embedding space for word tokens (default=256).
            hidden_dim: Hidden state size of the LSTM (default=256).
            num_layers: Number of LSTM layers (default=1).
        """
        super(SCSTModel, self).__init__()
        
        # CNN for image feature extraction
        self.cnn = cnn_model
        
        # Embedding layer for the captions
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM for sequence generation
        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        
        # Fully connected layers for combining image features with caption embeddings and generating final output
        self.fc_input = nn.Linear(512 + embedding_dim, embedding_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # Critic network for evaluating sequences
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, images, captions):
        """
        Forward pass for SCST to generate captions.

        Args:
            images: Tensor of images of shape (batch_size, channels, height, width).
            captions: Tensor of tokenized captions of shape (batch_size, seq_length).

        Returns:
            outputs: Tensor of shape (batch_size, seq_length, vocab_size) containing predicted word scores.
        """
        # Extract image features from CNN
        features = self.cnn(images).unsqueeze(1)  # Shape: (batch_size, 1, 512)

        # Embed the input captions
        captions_embed = self.embedding(captions)  # Shape: (batch_size, seq_length, embedding_dim)

        # Repeat image features for each caption word
        features = features.repeat(1, captions_embed.size(1), 1)  # Shape: (batch_size, seq_length, 512)
        
        # Concatenate image features with caption embeddings
        inputs = torch.cat((features, captions_embed), dim=2)  # Shape: (batch_size, seq_length, 512 + embedding_dim)
        inputs = self.fc_input(inputs)  # Shape: (batch_size, seq_length, embedding_dim)
        
        # Pass concatenated inputs through LSTM
        outputs, _ = self.rnn(inputs)
        
        # Generate final word predictions
        outputs = self.fc(outputs)  # Shape: (batch_size, seq_length, vocab_size)
        
        return outputs

    def evaluate(self, images, captions):
        """
        Evaluates captions and returns a score using the critic network.

        Args:
            images: Tensor of images.
            captions: Tensor of captions.

        Returns:
            Critic scores for the sequence.
        """
        # Forward pass similar to the main forward function
        features = self.cnn(images).unsqueeze(1)  # Shape: (batch_size, 1, 512)
        captions_embed = self.embedding(captions)  # Shape: (batch_size, seq_length, embedding_dim)
        features = features.repeat(1, captions_embed.size(1), 1)  # Shape: (batch_size, seq_length, 512)
        inputs = torch.cat((features, captions_embed), dim=2)  # Shape: (batch_size, seq_length, 512 + embedding_dim)
        inputs = self.fc_input(inputs)  # Shape: (batch_size, seq_length, embedding_dim)
        outputs, _ = self.rnn(inputs)
        
        # Use the critic network to evaluate the final hidden state
        return self.critic(outputs[:, -1, :])  # Shape: (batch_size, 1)

class SATModel(nn.Module):
    def __init__(self, cnn_model, vocab_size, embedding_dim=256, hidden_dim=256, num_layers=1):
        """
        Initializes the Show, Attend, and Tell (SAT) model, which includes an attention mechanism.

        Args:
            cnn_model: Pretrained CNN model for image feature extraction.
            vocab_size: Size of the vocabulary for the captions.
            embedding_dim: Dimension of the embedding space for word tokens (default=256).
            hidden_dim: Hidden state size of the LSTM (default=256).
            num_layers: Number of LSTM layers (default=1).
        """
        super(SATModel, self).__init__()
        
        # CNN for image feature extraction
        self.cnn = cnn_model
        
        # Embedding layer for the captions
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM for generating sequences
        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        
        # Fully connected layers for combining image features with caption embeddings and generating final output
        self.fc_input = nn.Linear(hidden_dim + embedding_dim, embedding_dim)
        self.fc = nn.Linear(embedding_dim, vocab_size)
        
        # Attention mechanism to focus on specific parts of the image
        self.attention_fc = nn.Linear(512, embedding_dim)

    def forward(self, images, captions):
        """
        Forward pass with attention for generating captions.

        Args:
            images: Tensor of images of shape (batch_size, channels, height, width).
            captions: Tensor of tokenized captions of shape (batch_size, seq_length).

        Returns:
            outputs: Tensor of shape (batch_size, seq_length, vocab_size) containing predicted word scores.
        """
        # Extract image features
        features = self.cnn(images)  # Shape: (batch_size, 512)
        features = features.unsqueeze(1)  # Shape: (batch_size, 1, 512)

        # Embed the input captions
        captions_embed = self.embedding(captions)  # Shape: (batch_size, seq_length, embedding_dim)

        # Apply attention to image features
        attended_features = self.attention_fc(features)  # Shape: (batch_size, 1, embedding_dim)
        attended_features = attended_features.repeat(1, captions_embed.size(1), 1)  # Shape: (batch_size, seq_length, embedding_dim)

        # Concatenate attended features with caption embeddings
        inputs = torch.cat((attended_features, captions_embed), dim=2)  # Shape: (batch_size, seq_length, hidden_dim + embedding_dim)
        inputs = self.fc_input(inputs)  # Shape: (batch_size, seq_length, embedding_dim)

        # Pass inputs through LSTM
        outputs, _ = self.rnn(inputs)
        
        # Generate final word predictions
        outputs = self.fc(outputs)  # Shape: (batch_size, seq_length, vocab_size)
        
        return outputs

# Load a pretrained ResNet18 model and remove the final fully connected layer to extract features
cnn_model = models.resnet18(pretrained=True)
cnn_model.fc = nn.Identity()  # Remove the classification layer

# Define the vocabulary size
vocab_size = len(word_to_idx)

# Initialize models with the modified CNN backbone and appropriate parameters
nic_model = ImageCaptioningModel(cnn_model, vocab_size, embedding_dim=256, hidden_dim=256)
scst_model = SCSTModel(cnn_model, vocab_size, embedding_dim=256, hidden_dim=256)
sat_model = SATModel(cnn_model, vocab_size, embedding_dim=256, hidden_dim=256)

# Set device to CPU (can be changed to 'cuda' if GPU is available)
device = 'cpu'
nic_model = nic_model.to(device)
scst_model = scst_model.to(device)
sat_model = sat_model.to(device)

# Define the loss function (CrossEntropy) and ignore <PAD> tokens during loss calculation
criterion = nn.CrossEntropyLoss(ignore_index=word_to_idx['<PAD>'])

# Define optimizers for each model (using Adam with learning rate 0.001)
optimizer_nic = optim.Adam(nic_model.parameters(), lr=0.001)
optimizer_scst = optim.Adam(scst_model.parameters(), lr=0.001)
optimizer_sat = optim.Adam(sat_model.parameters(), lr=0.001)

def compute_rouge_scores(hypothesis, references):
    """
    Computes ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L) between a hypothesis and references.

    Args:
        hypothesis: The generated hypothesis caption as a string.
        references: A list of reference captions to compare against.

    Returns:
        A dictionary with the averaged ROUGE-1, ROUGE-2, and ROUGE-L F1-scores.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
    num_references = len(references)
    
    # Compare the hypothesis against each reference and accumulate scores
    for ref in references:
        score = scorer.score(ref, hypothesis)
        for key in scores:
            scores[key] += score[key].fmeasure  # Accumulate F1 scores
    
    # Average the scores across all references
    for key in scores:
        scores[key] /= num_references
    
    return scores

def compute_bleu_scores(hypotheses, references):
    """
    Computes the BLEU score between a set of hypotheses and references.

    Args:
        hypotheses: A list of generated hypotheses as strings.
        references: A list of lists where each element contains reference captions for a hypothesis.

    Returns:
        Mean BLEU score across all hypotheses.
    """
    smooth = SmoothingFunction().method4  # Smoothing method to handle edge cases in BLEU
    bleu_scores = []
    
    # Compute BLEU score for each hypothesis-reference pair
    for hyp, refs in zip(hypotheses, references):
        refs = [ref.split() for ref in refs]  # BLEU expects a list of list of reference tokens
        hyp = hyp.split()  # Split the hypothesis into tokens
        score = sentence_bleu(refs, hyp, smoothing_function=smooth)
        bleu_scores.append(score)
    
    # Return the mean BLEU score across all samples
    return np.mean(bleu_scores)

def evaluate_model_with_rouge_and_bleu(model, dataloader, word_to_idx, idx_to_word):
    """
    Evaluates a model using both ROUGE and BLEU metrics on a given dataloader.

    Args:
        model: The captioning model to evaluate.
        dataloader: DataLoader for the validation or test set.
        word_to_idx: Dictionary mapping words to indices.
        idx_to_word: Dictionary mapping indices back to words.

    Returns:
        rouge_scores: Averaged ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L) for the dataset.
        bleu_score: Averaged BLEU score for the dataset.
    """
    model.eval()  # Set the model to evaluation mode
    all_hypotheses = []  # Store generated captions (hypotheses)
    all_references = []  # Store ground truth captions (references)

    with torch.no_grad():
        for images, captions in dataloader:
            images = images.to(device)
            captions = captions.to(device)

            # Pass the images and captions through the model to get predictions
            outputs = model(images, captions)

            # Get the most likely word predictions from the output
            _, predicted = torch.max(outputs, dim=2)
            predicted = predicted.cpu().numpy()
            captions = captions.cpu().numpy()

            # Detokenize both the predicted and ground truth captions
            for i in range(predicted.shape[0]):
                hyp = detokenize_caption(predicted[i], idx_to_word)  # Hypothesis (predicted)
                ref = detokenize_caption(captions[i], idx_to_word)   # Reference (ground truth)
                all_hypotheses.append(hyp)
                all_references.append(ref)
    
    # Compute ROUGE and BLEU scores using the accumulated hypotheses and references
    rouge_scores = compute_rouge_scores(' '.join(all_hypotheses), all_references)
    bleu_score = compute_bleu_scores(all_hypotheses, all_references)
    
    return rouge_scores, bleu_score

def train_and_validate_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs=5, patience=3):
    """
    Trains and validates the model while monitoring the loss and triggering early stopping if needed.

    Args:
        model: The image captioning model to be trained.
        train_dataloader: DataLoader for the training set.
        val_dataloader: DataLoader for the validation set.
        criterion: Loss function (CrossEntropyLoss).
        optimizer: Optimizer (Adam).
        num_epochs: Maximum number of epochs for training (default: 5).
        patience: Number of epochs to wait before early stopping if no improvement (default: 3).

    Returns:
        train_losses: List of average training losses per epoch.
        val_losses: List of validation losses per epoch.
    """
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        # Training loop
        for batch_idx, (images, captions) in enumerate(train_dataloader):
            images = images.to(device)
            captions = captions.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, captions)
            loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        train_losses.append(avg_epoch_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}")

        # Validation step
        val_loss = evaluate_model_on_val_set(model, val_dataloader, criterion)
        val_losses.append(val_loss)
        print(f"Validation Loss: {val_loss:.4f}")

        # Check for improvement and early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

    return train_losses, val_losses

def evaluate_model_on_val_set(model, val_dataloader, criterion):
    """
    Evaluates the model on the validation set.

    Args:
        model: The image captioning model to be evaluated.
        val_dataloader: DataLoader for the validation set.
        criterion: Loss function (CrossEntropyLoss).

    Returns:
        Average validation loss across the validation set.
    """
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for images, captions in val_dataloader:
            images = images.to(device)
            captions = captions.to(device)
            outputs = model(images, captions)
            loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
            total_loss += loss.item()

    return total_loss / len(val_dataloader)

def evaluate_model_with_attention(model, dataloader, word_to_idx, idx_to_word):
    """
    Evaluates the model using attention mechanism with ROUGE and BLEU scores.

    Args:
        model: The attention-based image captioning model.
        dataloader: DataLoader for the test set.
        word_to_idx: Mapping of words to indices.
        idx_to_word: Mapping of indices back to words.

    Returns:
        rouge_scores: ROUGE scores for the dataset.
        bleu_score: BLEU score for the dataset.
    """
    model.eval()
    all_hypotheses = []
    all_references = []

    with torch.no_grad():
        for images, captions in dataloader:
            images = images.to(device)
            captions = captions.to(device)
            outputs = model(images, captions)

            # Get the predicted tokens
            _, predicted = torch.max(outputs, dim=2)
            predicted = predicted.cpu().numpy()
            captions = captions.cpu().numpy()

            for i in range(predicted.shape[0]):
                hyp = detokenize_caption(predicted[i], idx_to_word)
                ref = detokenize_caption(captions[i], idx_to_word)
                all_hypotheses.append(hyp)
                all_references.append(ref)

    # Compute ROUGE and BLEU scores
    rouge_scores = compute_rouge_scores(' '.join(all_hypotheses), all_references)
    bleu_score = compute_bleu_scores(all_hypotheses, all_references)
    
    return rouge_scores, bleu_score

def evaluate_and_save_metrics(model, dataloader, word_to_idx, idx_to_word, model_type):
    """
    Evaluates the model using ROUGE and BLEU, and saves the metrics for analysis.

    Args:
        model: The image captioning model to evaluate.
        dataloader: DataLoader for the test set.
        word_to_idx: Mapping of words to indices.
        idx_to_word: Mapping of indices back to words.
        model_type: Type of model ('nic', 'scst', etc.) used for metric tracking.

    Returns:
        None
    """
    # Compute ROUGE and BLEU scores
    rouge_scores, bleu_score = evaluate_model_with_rouge_and_bleu(model, dataloader, word_to_idx, idx_to_word)

    # Append results to corresponding lists based on model type
    if model_type == 'nic':
        nic_rouge_scores.append(rouge_scores)
        nic_bleu_score.append(bleu_score)
    elif model_type == 'scst':
        scst_rouge_scores.append(rouge_scores)
        scst_bleu_score.append(bleu_score)

    # Additional metrics: lengths of captions and saving captions
    all_lengths = []
    all_captions = []

    with torch.no_grad():
        for images, captions in dataloader:
            captions = captions.cpu().numpy()
            for i in range(captions.shape[0]):
                caption_length = np.sum(captions[i] != word_to_idx['<PAD>'])
                all_lengths.append(caption_length)
                all_captions.append(' '.join([idx_to_word[idx] for idx in captions[i] if idx != word_to_idx['<PAD>']]))

    # Save results based on model type
    if model_type == 'nic':
        nic_caption_lengths.append(all_lengths)
        captions_nic.extend(all_captions)
    elif model_type == 'scst':
        scst_caption_lengths.append(all_lengths)
        captions_scst.extend(all_captions)

def display_image_and_captions(image, original_caption, predicted_caption):
    """
    Displays an image along with its original and predicted captions.

    Args:
        image (PIL.Image or ndarray): The image to display.
        original_caption (str): The original caption of the image.
        predicted_caption (str): The predicted caption generated by the model.
    """
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 1, 1)
    plt.imshow(image)  # Display the image
    plt.title(f"Predicted Caption: {predicted_caption}")  # Set the title with the predicted caption
    plt.axis('off')  # Turn off axis for a cleaner image display
    plt.show()

# Function to untransform the image (convert back from tensor)
def untransform_image(image_tensor, transform):
    """
    Untransforms an image tensor by reversing normalization and converting it to a PIL image.

    Args:
        image_tensor (torch.Tensor): The image tensor to be untransformed.
        transform (transforms.Compose): The original transform applied to the image.

    Returns:
        PIL.Image: The untransformed image in PIL format.
    """
    # Unnormalize the image (reverse normalization)
    unnormalize_transform = transforms.Compose([
        transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], 
                             std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
    ])
    
    image_tensor = unnormalize_transform(image_tensor)  # Apply the unnormalization

    # Clamp the tensor values to be within [0, 1] to make it valid for image display
    image_tensor = torch.clamp(image_tensor, 0, 1)
    
    # Convert the tensor to a PIL image for display
    return transforms.ToPILImage()(image_tensor)

def evaluate_model_and_show_predictions(model, dataloader, idx_to_word, device, transform):
    """
    Evaluates the model on a batch of data and displays an image with its predicted and original captions.

    Args:
        model (nn.Module): The trained image captioning model.
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
        idx_to_word (dict): Dictionary mapping word indices to actual words.
        device (str): Device on which to perform computation (e.g., 'cpu' or 'cuda').
        transform (transforms.Compose): The transformation used for preprocessing the images.
    """
    model.eval()  # Set the model to evaluation mode
    
    with torch.no_grad():  # Disable gradient computation
        # Iterate through the dataloader to get a batch of images and captions
        for images, captions in dataloader:
            images = images.to(device)  # Move images to the specified device
            captions = captions.to(device)  # Move captions to the specified device
            
            # Get predictions for the first image in the batch
            outputs = model(images, captions)  # Pass both images and captions to the model
            _, predicted = torch.max(outputs, dim=2)  # Get the index of the maximum predicted word
            predicted_caption = predicted[0].cpu().numpy()  # Convert the predicted tensor to a NumPy array
            
            # Convert captions (both original and predicted) to human-readable text
            original_caption_text = detokenize_caption(captions[0].cpu().numpy(), idx_to_word)
            predicted_caption_text = detokenize_caption(predicted_caption, idx_to_word)
            
            # Untransform the image tensor to convert it back to a displayable format (PIL image)
            image = untransform_image(images[0].cpu(), transform)
            
            # Display the image alongside the original and predicted captions
            display_image_and_captions(image, original_caption_text, predicted_caption_text)
            break  # Stop after displaying one image (for this example)

def convert_to_serializable(data):
    """
    Recursively converts data types that are not JSON-serializable (e.g., NumPy data types) 
    to their native Python equivalents, making them suitable for JSON serialization.

    Args:
        data: The data to be converted. This can be a dictionary, list, NumPy array, scalar, or any other structure.

    Returns:
        The input data with all NumPy arrays converted to lists, and NumPy scalars converted to native Python types.
    """
    if isinstance(data, dict):
        # Recursively convert the dictionary's values
        return {k: convert_to_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        # Recursively convert each item in the list
        return [convert_to_serializable(i) for i in data]
    elif isinstance(data, np.ndarray):
        # Convert NumPy arrays to Python lists
        return data.tolist()
    elif isinstance(data, np.generic):
        # Convert NumPy scalar types to native Python scalars
        return data.item()
    else:
        # Return the data as is if no conversion is needed
        return data


def save_metrics_to_json(filename):
    """
    Saves training and evaluation metrics to a JSON file for later analysis.

    The function gathers global metrics (losses, scores, caption lengths) and converts
    them to JSON-compatible format using the `convert_to_serializable` function.

    Args:
        filename (str): The path and name of the JSON file to save the metrics.
    """
    # Prepare the data structure with all the metrics to be saved
    data = {
        'epoch_losses': epoch_losses,  # List of training loss values per epoch
        'val_losses': val_losses,  # List of validation loss values per epoch
        'nic_rouge_scores': nic_rouge_scores,  # ROUGE scores for the NIC model
        'scst_rouge_scores': scst_rouge_scores,  # ROUGE scores for the SCST model
        'nic_bleu_score': nic_bleu_score,  # BLEU scores for the NIC model
        'scst_bleu_score': scst_bleu_score,  # BLEU scores for the SCST model
        'nic_caption_lengths': nic_caption_lengths,  # Lengths of NIC-generated captions
        'scst_caption_lengths': scst_caption_lengths,  # Lengths of SCST-generated captions
        'captions_nic': captions_nic,  # Generated captions from the NIC model
        'captions_scst': captions_scst  # Generated captions from the SCST model
    }
    
    # Convert the data into a JSON-serializable format using `convert_to_serializable`
    serializable_data = convert_to_serializable(data)
    
    # Write the serialized data to the specified JSON file
    with open(filename, 'w') as f:
        json.dump(serializable_data, f, indent=4)  # Indent the JSON for readability

# Example usage: Save the metrics to 'metrics1.json'
save_metrics_to_json('metrics1.json')


def train_and_evaluate_all_models(nic_model, scst_model, sat_model, train_dataloader, val_dataloader, criterion, optimizer_nic, optimizer_scst, optimizer_sat, word_to_idx, idx_to_word, num_epochs=5, patience=3):
    """
    Trains and evaluates three models: NIC, SCST, and SAT. Each model is trained, validated, 
    and evaluated using the BLEU and ROUGE metrics, and their results are saved.

    Args:
        nic_model: The NIC (Neural Image Captioning) model instance.
        scst_model: The SCST (Self-Critical Sequence Training) model instance.
        sat_model: The SAT (Show Attend and Tell) model instance.
        train_dataloader: DataLoader for training data.
        val_dataloader: DataLoader for validation data.
        criterion: Loss function used for training.
        optimizer_nic: Optimizer for the NIC model.
        optimizer_scst: Optimizer for the SCST model.
        optimizer_sat: Optimizer for the SAT model.
        word_to_idx: Dictionary mapping words to their index values.
        idx_to_word: Dictionary mapping indices back to words.
        num_epochs: Number of epochs to train each model.
        patience: Number of epochs to wait before early stopping if no improvement is seen.
    """
    
    # Train NIC model
    print("Training NIC model...")
    train_and_validate_model(nic_model, train_dataloader, val_dataloader, criterion, optimizer_nic, num_epochs=num_epochs, patience=patience)
    
    # Load the best NIC model from training and evaluate it
    print("Evaluating NIC model...")
    nic_model.load_state_dict(torch.load('best_model.pth'))
    evaluate_and_save_metrics(nic_model, val_dataloader, word_to_idx, idx_to_word, model_type='nic')
    
    # Train SCST model
    print("Training SCST model...")
    train_and_validate_model(scst_model, train_dataloader, val_dataloader, criterion, optimizer_scst, num_epochs=num_epochs, patience=patience)
    
    # Load the best SCST model from training and evaluate it
    print("Evaluating SCST model...")
    scst_model.load_state_dict(torch.load('best_model.pth'))
    evaluate_and_save_metrics(scst_model, val_dataloader, word_to_idx, idx_to_word, model_type='scst')
    
    # Train SAT model
    print("Training SAT model...")
    train_and_validate_model(sat_model, train_dataloader, val_dataloader, criterion, optimizer_sat, num_epochs=num_epochs, patience=patience)
    
    # Load the best SAT model from training and evaluate it
    print("Evaluating SAT model...")
    sat_model.load_state_dict(torch.load('best_model.pth'))
    evaluate_and_save_metrics(sat_model, val_dataloader, word_to_idx, idx_to_word, model_type='sat')


def main():
    """
    Main function that sets up models, dataloaders, and optimizers, and trains and evaluates
    three different models: NIC, SCST, and SAT.
    """
    # Assuming all required variables (models, dataloaders, criterion, optimizers) are already initialized
    
    # Train and evaluate all models (NIC, SCST, SAT)
    train_and_evaluate_all_models(nic_model, scst_model, sat_model, train_dataloader, val_dataloader, criterion, optimizer_nic, optimizer_scst, optimizer_sat, word_to_idx, idx_to_word)


# Example usage - Call the main function to train and evaluate the models
if __name__ == "__main__":
    main()
