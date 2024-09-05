# Core PyTorch libraries
import torch  # Core library for tensor operations and deep learning
import torch.nn as nn  # Module containing neural network layers and functions
import torch.optim as optim  # Module for optimization algorithms

# PyTorch data utilities
from torch.utils.data import DataLoader, Subset  # Tools for managing and loading datasets

# Vision-related libraries
from torchvision import datasets, transforms, models  # Image datasets, transformations, and pre-trained models
from PIL import Image  # Image processing library
from PIL import ImageTk  # Tkinter support for displaying images

# COCO dataset tools
from pycocotools.coco import COCO  # API for interacting with COCO dataset annotations

# Numerical and time utilities
import numpy as np  # Library for numerical operations on arrays
import time  # Functions for measuring and managing time

# OS and system utilities
import os  # Utilities for interacting with the operating system
import psutil  # System and process monitoring tools (CPU, memory usage, etc.)

# Data visualization libraries
import matplotlib.pyplot as plt  # Plotting library for creating visualizations
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # Backend for embedding Matplotlib plots in Tkinter
import seaborn as sns  # Statistical data visualization library

# Progress and GUI utilities
from tqdm import tqdm  # Progress bar for loops and iterations
import tkinter as tk  # Standard GUI library for Python
from tkinter import ttk, messagebox, filedialog  # Additional Tkinter modules for widgets, dialogs, and file handling

# Evaluation metrics
from rouge_score import rouge_scorer  # ROUGE score calculation for evaluating text summarization
from nltk.translate.bleu_score import corpus_bleu  # BLEU score calculation for evaluating machine translation

# Additional utility
from collections import Counter  # For counting hashable objects and creating frequency distributions


# Set seaborn style for plots
sns.set(style="whitegrid")

# Global list to store loss values for plotting
epoch_losses = []

def build_coco_vocab(coco, min_freq=5):
    """
    Build a vocabulary from COCO captions.

    Args:
        coco (COCO): COCO dataset object.
        min_freq (int): Minimum frequency threshold for including words in the vocabulary.

    Returns:
        tuple: Two dictionaries - word_to_idx and idx_to_word for vocabulary mapping.
    """
    ann_ids = coco.getAnnIds()
    anns = coco.loadAnns(ann_ids)
    
    # Extract and tokenize captions
    captions = [ann['caption'] for ann in anns]
    tokenized_captions = [caption.lower().split() for caption in captions]
    
    # Count word frequencies
    word_counter = Counter(word for caption in tokenized_captions for word in caption)
    
    # Build vocabulary based on frequency
    vocab = [word for word, freq in word_counter.items() if freq >= min_freq]
    vocab = ['<PAD>', '<START>', '<END>', '<UNK>'] + vocab
    
    # Create word to index and index to word mappings
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    
    return word_to_idx, idx_to_word

def tokenize_caption(caption, word_to_idx, max_length=20):
    """
    Convert a caption to a tensor of token indices.

    Args:
        caption (str): The caption text.
        word_to_idx (dict): Dictionary mapping words to indices.
        max_length (int): Maximum length of the tokenized caption.

    Returns:
        torch.Tensor: Tensor containing token indices.
    """
    tokens = [word_to_idx.get(word, word_to_idx['<UNK>']) for word in caption.lower().split()]
    tokens = tokens[:max_length]
    tokens += [word_to_idx['<PAD>']] * (max_length - len(tokens))
    return torch.tensor(tokens, dtype=torch.long)

def detokenize_caption(tokens, idx_to_word):
    """
    Convert a tensor of token indices back to a caption string.

    Args:
        tokens (list of int): List of token indices.
        idx_to_word (dict): Dictionary mapping indices to words.

    Returns:
        str: The detokenized caption string.
    """
    caption = []
    for token in tokens:
        word = idx_to_word.get(token, '')
        if word in ['<PAD>', '<START>', '<END>', '<UNK>']:
            continue
        caption.append(word)
    return ' '.join(caption)

def coco_collate_fn(batch, word_to_idx, max_length=20):
    """
    Custom collate function for DataLoader to handle COCO dataset batches.

    Args:
        batch (list of tuples): List of (image, caption) tuples.
        word_to_idx (dict): Dictionary mapping words to indices.
        max_length (int): Maximum length of the tokenized caption.

    Returns:
        tuple: A tuple containing batched images and captions.
    """
    images = []
    captions = []
    
    for image, caption in batch:
        images.append(image)
        tokenized_caption = tokenize_caption(caption[0], word_to_idx, max_length)
        captions.append(tokenized_caption)
    
    images = torch.stack(images, dim=0)
    captions = torch.stack(captions, dim=0)
    return images, captions

# COCO dataset root directory
coco_root = '/mnt/c/Users/muham/Downloads/coco7/'

# Define image transformations for COCO dataset
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128 pixels
    transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
    transforms.RandomRotation(15),  # Randomly rotate images by up to 15 degrees
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with precomputed means and stds
])

# Initialize COCO dataset
coco_train = COCO(f"{coco_root}/annotations/captions_train2017.json")
word_to_idx, idx_to_word = build_coco_vocab(coco_train)

def create_subset(dataset, max_size=100):
    """
    Create a random subset of the dataset.

    Args:
        dataset (Dataset): Dataset to create a subset from.
        max_size (int): Maximum number of samples in the subset.

    Returns:
        Subset: A subset of the original dataset.
    """
    indices = np.random.choice(len(dataset), max_size, replace=False)
    indices = list(map(int, indices))
    return Subset(dataset, indices)

def load_train_dataset(max_size):
    """
    Load and preprocess the COCO training dataset with a subset of data.

    Args:
        max_size (int): Maximum number of samples to include in the subset.

    Returns:
        Subset: A subset of the COCO training dataset.
    """
    train_dataset = datasets.CocoCaptions(
        root=f"{coco_root}/train2017",
        annFile=f"{coco_root}/annotations/captions_train2017.json",
        transform=transform
    )
    return create_subset(train_dataset, max_size)

class ImageCaptioningModel(nn.Module):
    """
    A PyTorch model for image captioning that combines a CNN for feature extraction
    with an LSTM for generating captions.

    Args:
        cnn_model (nn.Module): Pretrained CNN model for feature extraction.
        vocab_size (int): Size of the vocabulary.
        embedding_dim (int): Dimension of the embedding vector.
        hidden_dim (int): Dimension of the hidden state in the LSTM.
        num_layers (int): Number of layers in the LSTM.
    """
    def __init__(self, cnn_model, vocab_size, embedding_dim=256, hidden_dim=256, num_layers=1):
        super(ImageCaptioningModel, self).__init__()
        self.cnn = cnn_model  # CNN for feature extraction
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # Embedding layer
        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)  # LSTM for caption generation
        
        self.fc_input = nn.Linear(512 + embedding_dim, embedding_dim)  # Linear layer to process concatenated features and embeddings
        self.fc = nn.Linear(hidden_dim, vocab_size)  # Linear layer for output logits
    
    def forward(self, images, captions):
        """
        Forward pass of the model.

        Args:
            images (torch.Tensor): Batch of images.
            captions (torch.Tensor): Batch of caption indices.

        Returns:
            torch.Tensor: Output logits for each word in the captions.
        """
        # Extract features from images
        features = self.cnn(images).unsqueeze(1)  # Add sequence dimension
        
        # Embed captions
        captions_embed = self.embedding(captions)
        
        # Repeat features to match the sequence length
        features = features.repeat(1, captions_embed.size(1), 1)
        
        # Concatenate features and embeddings
        inputs = torch.cat((features, captions_embed), dim=2)
        inputs = self.fc_input(inputs)  # Process inputs
        
        # Pass through LSTM
        outputs, _ = self.rnn(inputs)
        outputs = self.fc(outputs)  # Get logits for each word
        return outputs

class SCSTModel(nn.Module):
    """
    A PyTorch model for image captioning with Self-Critical Sequence Training (SCST) 
    that combines a CNN for feature extraction with an LSTM for generating captions
    and includes a critic network.

    Args:
        cnn_model (nn.Module): Pretrained CNN model for feature extraction.
        vocab_size (int): Size of the vocabulary.
        embedding_dim (int): Dimension of the embedding vector.
        hidden_dim (int): Dimension of the hidden state in the LSTM.
        num_layers (int): Number of layers in the LSTM.
    """
    def __init__(self, cnn_model, vocab_size, embedding_dim=256, hidden_dim=256, num_layers=1):
        super(SCSTModel, self).__init__()
        self.cnn = cnn_model  # CNN for feature extraction
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # Embedding layer
        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)  # LSTM for caption generation
        
        self.fc_input = nn.Linear(512 + embedding_dim, embedding_dim)  # Linear layer to process concatenated features and embeddings
        self.fc = nn.Linear(hidden_dim, vocab_size)  # Linear layer for output logits
        self.critic = nn.Linear(hidden_dim, 1)  # Critic network for evaluating captions

    def forward(self, images, captions):
        """
        Forward pass of the SCST model.

        Args:
            images (torch.Tensor): Batch of images.
            captions (torch.Tensor): Batch of caption indices.

        Returns:
            torch.Tensor: Output logits for each word in the captions.
        """
        # Extract features from images
        features = self.cnn(images).unsqueeze(1)  # Add sequence dimension
        
        # Embed captions
        captions_embed = self.embedding(captions)
        
        # Repeat features to match the sequence length
        features = features.repeat(1, captions_embed.size(1), 1)
        
        # Concatenate features and embeddings
        inputs = torch.cat((features, captions_embed), dim=2)
        inputs = self.fc_input(inputs)  # Process inputs
        
        # Pass through LSTM
        outputs, _ = self.rnn(inputs)
        outputs = self.fc(outputs)  # Get logits for each word
        return outputs

    def evaluate(self, images, captions):
        """
        Evaluate the captions using the critic network.

        Args:
            images (torch.Tensor): Batch of images.
            captions (torch.Tensor): Batch of caption indices.

        Returns:
            torch.Tensor: Critic scores for the generated captions.
        """
        features = self.cnn(images).unsqueeze(1)
        captions_embed = self.embedding(captions)
        features = features.repeat(1, captions_embed.size(1), 1)
        inputs = torch.cat((features, captions_embed), dim=2)
        inputs = self.fc_input(inputs)
        outputs, _ = self.rnn(inputs)
        return self.critic(outputs[:, -1, :])  # Evaluate the last word in the sequence

# Initialize CNN model with ResNet18
cnn_model = models.resnet18(pretrained=True)
cnn_model.fc = nn.Identity()  # Remove the final classification layer

# Initialize the image captioning models
vocab_size = len(word_to_idx)
nic_model = ImageCaptioningModel(cnn_model, vocab_size, embedding_dim=256, hidden_dim=256)
scst_model = SCSTModel(cnn_model, vocab_size, embedding_dim=256, hidden_dim=256)

# Set device for models
device = 'cpu'
nic_model = nic_model.to(device)
scst_model = scst_model.to(device)

# Define loss function and optimizers
criterion = nn.CrossEntropyLoss(ignore_index=word_to_idx['<PAD>'])
optimizer_nic = optim.Adam(nic_model.parameters(), lr=0.001)
optimizer_scst = optim.Adam(scst_model.parameters(), lr=0.001)

def compute_rouge_scores(hypothesis, references):
    """
    Compute ROUGE scores for evaluating text generation.

    Args:
        hypothesis (str): Generated text.
        references (list of str): List of reference texts.

    Returns:
        dict: ROUGE-1, ROUGE-2, and ROUGE-L scores.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
    for ref in references:
        score = scorer.score(ref, hypothesis)
        for key in scores:
            scores[key] += score[key].fmeasure
    num_references = len(references)
    for key in scores:
        scores[key] /= num_references
    return scores


def compute_bleu_scores(hypotheses, references):
    """
    Compute BLEU score for evaluating text generation.

    Args:
        hypotheses (list of str): List of generated texts.
        references (list of str): List of reference texts.

    Returns:
        float: BLEU score.
    """
    # Tokenize the references and hypotheses
    references = [[ref.split()] for ref in references]  # Each reference needs to be a list of token lists
    hypotheses = [hyp.split() for hyp in hypotheses]
    
    # Compute BLEU score
    bleu_score = corpus_bleu(references, hypotheses)
    return bleu_score

def evaluate_model_with_rouge(model, dataloader, word_to_idx, idx_to_word, progress_var, progress_label, rouge_label, start_time):
    """
    Evaluate a model using ROUGE scores and update GUI with progress.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader for the evaluation data.
        word_to_idx (dict): Mapping from words to indices.
        idx_to_word (dict): Mapping from indices to words.
        progress_var (tk.Variable): Progress bar variable.
        progress_label (tk.Label): Label for progress display.
        rouge_label (tk.Label): Label for ROUGE scores.
        start_time (float): Start time of the evaluation.
    """
    model.eval()
    hypotheses = []
    references = []
    
    total_batches = len(dataloader)
    
    with torch.no_grad():
        for batch_idx, (images, captions) in enumerate(dataloader):
            images = images.to(device)
            captions = captions.to(device)
            
            # Generate outputs
            outputs = model(images, captions)
            _, predicted = torch.max(outputs, 2)
            
            for i in range(images.size(0)):
                hyp = detokenize_caption(predicted[i].cpu().numpy(), idx_to_word)
                ref = detokenize_caption(captions[i].cpu().numpy(), idx_to_word)
                hypotheses.append(hyp)
                references.append([ref])
            
            elapsed_time = time.time() - start_time
            remaining_time = elapsed_time * (total_batches - (batch_idx + 1)) / (batch_idx + 1)

            # Update progress bar and label
            progress_var.set((batch_idx + 1) / total_batches * 100)
            progress_label.config(text=f"Evaluating: {batch_idx + 1}/{total_batches}, Time Remaining: {remaining_time:.2f}s")
            root.update_idletasks()

    # Compute ROUGE scores
    rouge_scores = compute_rouge_scores(hypotheses[0], references[0])
    
    # Update the GUI with the ROUGE scores
    rouge_label.config(text=f"ROUGE-1: {rouge_scores['rouge1']:.4f}, ROUGE-2: {rouge_scores['rouge2']:.4f}, ROUGE-L: {rouge_scores['rougeL']:.4f}")
    
    progress_label.config(text="Evaluation complete!")
    end_time = time.time()
    print(f"Evaluation Time: {end_time - start_time:.2f} seconds")

def train_model(model, dataloader, criterion, optimizer, num_epochs, progress_var, progress_label, start_time, canvas):
    """
    Train the specified model and update the progress in the GUI.

    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): DataLoader for the training dataset.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        num_epochs (int): Number of epochs to train.
        progress_var (tk.DoubleVar): Variable for updating the progress bar.
        progress_label (tk.Label): Label for displaying progress information.
        start_time (float): Start time of the training process.
        canvas (FigureCanvasTkAgg): Canvas for plotting the loss curve.
    """
    model.train()
    total_batches = len(dataloader)
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_idx, (images, captions) in enumerate(dataloader):
            images = images.to(device)
            captions = captions.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images, captions)
            
            # Compute loss
            loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

            # Update progress
            elapsed_time = time.time() - start_time
            remaining_time = elapsed_time * ((num_epochs * total_batches) - (epoch * total_batches + batch_idx + 1)) / (epoch * total_batches + batch_idx + 1)
            
            # Update progress bar and label
            progress_var.set(((batch_idx + 1) + epoch * total_batches) / (total_batches * num_epochs) * 100)
            progress_label.config(text=f"Training: Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{total_batches}, Time Remaining: {remaining_time:.2f}s")
            root.update_idletasks()

        avg_epoch_loss = epoch_loss / total_batches
        epoch_losses.append(avg_epoch_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}")

    plot_loss_curve(canvas)  # Plot losses after training

    report_resources()
    progress_label.config(text="Training complete!")
    messagebox.showinfo("Training", "Training complete!")

def plot_loss_curve(canvas):
    """
    Plot the training loss curve and update the GUI canvas.

    Args:
        canvas (FigureCanvasTkAgg): Canvas for plotting the loss curve.
    """
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(epoch_losses, label='Epoch Losses', marker='o')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Curve')
    ax.legend()
    ax.grid(True)

    # Update the canvas with the new plot
    canvas.figure = fig
    canvas.draw()

def report_resources():
    """
    Report the current CPU and memory usage.

    Returns:
        tuple: CPU usage percentage and memory usage percentage.
    """
    cpu_usage = psutil.cpu_percent()
    memory_info = psutil.virtual_memory()
    memory_usage = memory_info.percent
    return cpu_usage, memory_usage

def test_image_captioning(model, image_path, word_to_idx, idx_to_word):
    """
    Generate a caption for a given image using the specified model.

    Args:
        model (nn.Module): The trained model for image captioning.
        image_path (str): Path to the image file.
        word_to_idx (dict): Dictionary mapping words to indices.
        idx_to_word (dict): Dictionary mapping indices to words.

    Returns:
        str: Generated caption for the image.
    """
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Generate output
        outputs = model(image, torch.tensor([[word_to_idx['<START>']]], device=device))
        outputs = outputs.squeeze(0)
        _, predicted = torch.max(outputs, 1)
        
        # Convert prediction to caption
        caption = detokenize_caption(predicted.cpu().numpy(), idx_to_word)
        return caption

def start_gui():
    """
    Initialize and start the GUI for training, evaluation, and testing of image captioning models.
    """
    global root, epoch_losses, device, nic_model, scst_model, criterion, optimizer_nic, optimizer_scst, word_to_idx, idx_to_word, plot_visible
    
    root = tk.Tk()
    root.title("Image Captioning and Evaluation")
    root.geometry('800x800')

    # Configure GUI style
    style = ttk.Style()
    style.configure("TLabel", font=("Helvetica", 12))
    style.configure("TButton", font=("Helvetica", 12), padding=10)
    style.map("TButton",
              background=[('active', '#4CAF50')],
              foreground=[('active', '#FFFFFF')],
              relief=[('pressed', 'sunken')],
              bordercolor=[('focus', '#4CAF50')])

    # Create tabs for training, evaluation, and testing
    tab_control = ttk.Notebook(root)
    training_tab = ttk.Frame(tab_control)
    evaluation_tab = ttk.Frame(tab_control)
    test_tab = ttk.Frame(tab_control)

    tab_control.add(training_tab, text='Training')
    tab_control.add(evaluation_tab, text='Evaluation')
    tab_control.add(test_tab, text='Test')
    tab_control.pack(expand=1, fill='both')

    # Training Tab
    ttk.Label(training_tab, text="Training Settings", font=("Helvetica", 14)).pack(pady=10)

    ttk.Label(training_tab, text="Select Model:").pack(pady=5)
    model_selection = ttk.Combobox(training_tab, values=["NIC Model", "SCST Model"])
    model_selection.current(0)
    model_selection.pack(pady=5)
    
    ttk.Label(training_tab, text="Data Size:").pack(pady=5)
    data_size_entry = ttk.Entry(training_tab)
    data_size_entry.pack(pady=5)
    
    ttk.Label(training_tab, text="Number of Epochs:").pack(pady=5)
    num_epochs_entry = ttk.Entry(training_tab)
    num_epochs_entry.pack(pady=5)

    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(training_tab, orient="horizontal", length=400, mode="determinate", variable=progress_var)
    progress_bar.pack(pady=10)
    
    progress_label = ttk.Label(training_tab, text="Progress: 0%", font=("Helvetica", 12))
    progress_label.pack(pady=5)

    # Canvas for plotting loss curve
    loss_canvas_frame = tk.Frame(training_tab)
    loss_canvas_frame.pack(pady=10)
    
    fig, ax = plt.subplots(figsize=(5, 3))
    canvas = FigureCanvasTkAgg(fig, master=loss_canvas_frame)
    canvas.get_tk_widget().pack()

    plot_visible = True  # Flag to track visibility of the plot

    def toggle_plot_visibility():
        """
        Toggle the visibility of the plot in the GUI.
        """
        global plot_visible
        if plot_visible:
            canvas.get_tk_widget().pack_forget()
            plot_button.config(text="Show Plot")
        else:
            canvas.get_tk_widget().pack()
            plot_button.config(text="Hide Plot")
        plot_visible = not plot_visible

    plot_button = ttk.Button(training_tab, text="Hide Plot", command=toggle_plot_visibility)
    plot_button.pack(pady=5)

    def on_train_button_click():
        """
        Start the training process based on the selected model and training parameters.
        """
        global epoch_losses
        try:
            data_size = int(data_size_entry.get())
            num_epochs = int(num_epochs_entry.get())
            selected_model = model_selection.get()
            
            if data_size <= 0 or num_epochs <= 0:
                raise ValueError("Data size and number of epochs must be positive.")
            
            # Load dataset and start training
            dataloader = DataLoader(load_train_dataset(max_size=data_size), batch_size=4, collate_fn=lambda x: coco_collate_fn(x, word_to_idx))
            start_time = time.time()

            if selected_model == "NIC Model":
                train_model(nic_model, dataloader, criterion, optimizer_nic, num_epochs, progress_var, progress_label, start_time, canvas)
                torch.save(nic_model.state_dict(), 'nic_model.pth')
            elif selected_model == "SCST Model":
                train_model(scst_model, dataloader, criterion, optimizer_scst, num_epochs, progress_var, progress_label, start_time, canvas)
                torch.save(scst_model.state_dict(), 'scst_model.pth')

            cpu_usage, memory_usage = report_resources()
            messagebox.showinfo("Training Complete", f"Training complete!\nCPU Usage: {cpu_usage:.1f}%\nMemory Usage: {memory_usage:.1f}%")

        except ValueError as e:
            messagebox.showerror("Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", str(e))

    train_button = ttk.Button(training_tab, text="Start Training", command=on_train_button_click)
    train_button.pack(pady=10)

    # Evaluation Tab
    ttk.Label(evaluation_tab, text="Evaluation Settings", font=("Helvetica", 14)).pack(pady=10)

    ttk.Label(evaluation_tab, text="Select Model for Evaluation:").pack(pady=5)
    eval_model_selection = ttk.Combobox(evaluation_tab, values=["NIC Model", "SCST Model"])
    eval_model_selection.current(0)
    eval_model_selection.pack(pady=5)
    
    progress_var_eval = tk.DoubleVar()
    progress_bar_eval = ttk.Progressbar(evaluation_tab, orient="horizontal", length=400, mode="determinate", variable=progress_var_eval)
    progress_bar_eval.pack(pady=10)
    
    progress_label_eval = ttk.Label(evaluation_tab, text="Progress: 0%", font=("Helvetica", 12))
    progress_label_eval.pack(pady=5)

    rouge_label = ttk.Label(evaluation_tab, text="ROUGE Scores: N/A", font=("Helvetica", 12))
    rouge_label.pack(pady=5)

    def on_evaluate_button_click():
        """
        Start the evaluation process based on the selected model.
        """
        try:
            selected_eval_model = eval_model_selection.get()
            dataloader = DataLoader(load_train_dataset(max_size=100), batch_size=4, collate_fn=lambda x: coco_collate_fn(x, word_to_idx))
            start_time = time.time()

            if selected_eval_model == "NIC Model":
                nic_model.load_state_dict(torch.load('nic_model.pth'))
                evaluate_model_with_rouge(nic_model, dataloader, word_to_idx, idx_to_word, progress_var_eval, progress_label_eval, rouge_label, start_time)
            elif selected_eval_model == "SCST Model":
                scst_model.load_state_dict(torch.load('scst_model.pth'))
                evaluate_model_with_rouge(scst_model, dataloader, word_to_idx, idx_to_word, progress_var_eval, progress_label_eval, rouge_label, start_time)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    evaluate_button = ttk.Button(evaluation_tab, text="Evaluate Model", command=on_evaluate_button_click)
    evaluate_button.pack(pady=10)

    # Test Tab
    ttk.Label(test_tab, text="Test Image Captioning", font=("Helvetica", 14)).pack(pady=10)

    def on_load_model_button_click():
        """
        Load a pre-trained model from a file and set it for image captioning.

        This function opens a file dialog to select a model file, loads the model's state_dict,
        and updates the global `test_model` with the loaded weights.
        """
        global test_model
        model_path = filedialog.askopenfilename(filetypes=[("Model Files", "*.pth")])
        if not model_path:
            return

        # Initialize the model and load the state_dict
        test_model = ImageCaptioningModel(cnn_model, vocab_size, embedding_dim=256, hidden_dim=256)
        test_model.load_state_dict(torch.load(model_path))
        test_model.to(device)
        messagebox.showinfo("Model Loaded", "Model successfully loaded.")

    # Button to load the model
    load_model_button = ttk.Button(test_tab, text="Load Model", command=on_load_model_button_click)
    load_model_button.pack(pady=5)

    def on_load_image_button_click():
        """
        Load an image file and display it in the GUI.

        This function opens a file dialog to select an image file, resizes the image,
        and updates the image display label.
        """
        global image_path
        image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if not image_path:
            return

        # Load and resize the image
        image = Image.open(image_path).resize((300, 300))
        image_tk = ImageTk.PhotoImage(image)
        image_label.config(image=image_tk)
        image_label.image = image_tk

    # Button to load an image
    load_image_button = ttk.Button(test_tab, text="Load Image", command=on_load_image_button_click)
    load_image_button.pack(pady=5)

    def on_show_captions_button_click():
        """
        Generate and display a caption for the loaded image using the loaded model.

        This function uses the `test_image_captioning` function to generate a caption for the image
        and displays it in the caption label.
        """
        try:
            caption = test_image_captioning(test_model, image_path, word_to_idx, idx_to_word)
            caption_text.set(caption)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # Button to generate and display captions
    show_captions_button = ttk.Button(test_tab, text="Show Captions", command=on_show_captions_button_click)
    show_captions_button.pack(pady=5)

    # Label to display the loaded image
    image_label = ttk.Label(test_tab)
    image_label.pack(pady=10)

    # Label to display the generated caption
    caption_text = tk.StringVar()
    caption_label = ttk.Label(test_tab, textvariable=caption_text, wraplength=500)
    caption_label.pack(pady=10)

    # Resource Usage Display
    def update_resource_usage():
        """
        Update the displayed CPU and memory usage every second.

        This function retrieves resource usage information and updates the `resource_label`.
        It schedules itself to run again after 1000 milliseconds (1 second).
        """
        cpu_usage, memory_usage = report_resources()
        resource_label.config(text=f"CPU Usage: {cpu_usage:.1f}% | Memory Usage: {memory_usage:.1f}%")
        root.after(1000, update_resource_usage)

    # Label to display CPU and memory usage
    resource_label = ttk.Label(root, text="CPU Usage: N/A | Memory Usage: N/A", font=("Helvetica", 10))
    resource_label.pack(side=tk.BOTTOM, anchor='e', pady=10)

    update_resource_usage()

    # Start the GUI event loop
    root.mainloop()

# Start the GUI application
start_gui()
