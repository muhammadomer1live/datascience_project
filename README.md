# Advanced Image Captioning Techniques Using COCO Dataset: A Comparative Study

## Description
This project utilizes machine learning, data visualization, and natural language processing libraries in Python. It was developed on WSL 2.0 (Ubuntu for Windows 11) and is compatible with both Linux and Windows.

## Requirements

Before running the project, make sure you have the following software installed:

- **Python 3.x** (recommended version: Python 3.8 or later)
- **pip** (Python package installer)
- **virtualenv** (optional, but recommended for managing dependencies)
  
## Installation Instructions

### 1. Setting Up Environment (Linux and Windows)

#### Linux (Ubuntu or any distribution):
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. (Optional) Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies using `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   python main.py
   ```

#### Windows (Native):
1. Install **Python 3.x** from [Python's official website](https://www.python.org/downloads/).

2. Clone the repository using Git:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

3. (Optional) Create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Run the application:
   ```bash
   python main.py
   ```

#### Windows (WSL 2.0 on Ubuntu):
1. Ensure that WSL 2.0 is installed and configured correctly by following the instructions [here](https://docs.microsoft.com/en-us/windows/wsl/install).

2. Open WSL and navigate to your project directory:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

3. (Optional) Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Run the application:
   ```bash
   python main.py
   ```

### 2. Managing Dependencies
If you install or update any additional packages, make sure to update the `requirements.txt` file:
```bash
pip freeze > requirements.txt
```

## Running the Application
To run the application, navigate to the project directory and execute the following command:

```bash
python main.py
```

Ensure the required data files or resources are in the correct directories if the program depends on external data.

## Additional Notes
- **WSL 2.0** allows you to run Linux applications directly on Windows. If you experience issues with the display of graphical applications like `tkinter` or `matplotlib` under WSL, you may need an X server (such as [VcXsrv](https://sourceforge.net/projects/vcxsrv/)) to forward graphical output from WSL to Windows.

## Installed Libraries
   ```bash
pip install torch torchvision Pillow pycocotools numpy psutil matplotlib seaborn tqdm rouge-score nltk scikit-learn scipy
   ```
  
- For Windows users who wish to run the application natively, the `tkinter` and other graphical libraries should work as expected, but you may need to install additional dependencies using your package manager.

## License
[Specify your license here, e.g., MIT License]

---

This format will give clear instructions to users on how to set up and run the program in both Linux and Windows environments, including WSL 2.
