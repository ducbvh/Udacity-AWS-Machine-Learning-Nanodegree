# CAPSTONE PROJECT -  FACIAL DEOCCLUSION
## Udacity: AWS Machine Learning Engineer Nanodegree

## Overview
This project is part of the **Udacity AWS Machine Learning Capstone** course. It leverages AWS resources to build, train, and deploy machine learning models effectively. In this Project, I built my own dataset with face occlusion by face mask and try to train a model can deocclusion it.

### Project Documentation

Detailed documentation for this project is available in the following PDF files:

- **[Project Proposal](Proposal%20-%20Facial%20Deocculusion.pdf)**: Outlines the project objectives, methodology, and expected outcomes. Review link in the student submission notes [Review Proposal](https://learn.udacity.com/nanodegrees/nd189/parts/cd0549/lessons/a78d5b7d-95d4-4c8e-bcf6-16560e307a81/concepts/a78d5b7d-95d4-4c8e-bcf6-16560e307a81-submit-project?lesson_tab=lesson)
- **[Project Report](Report_Capstone_Project.pdf)**: Contains the project findings, results, and conclusions.

Please refer to these documents for a deeper understanding of the project's design, implementation, and analysis.


By following the instructions and reviewing the documentation, you'll gain insight into the project workflow and learn about the key challenges and solutions explored in this capstone.

## Table of Contents
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)

## Project Structure
The project is organized into the following files and directories:

- **`utils/`**: Directory containing utility scripts used throughout the project, such as data preprocessing, define models's class, model architecture, evaluation metrics, and helper functions.
- **`requirements.txt`**: Lists all necessary Python packages. 
- **`train.py`**: Main script for model training. It includes configurations for loading data, defining the model architecture, and setting training parameters.
- **`inference.py`**: Script for running inference on new data using the trained model. Supports various input formats.
- **`logs/`**: Directory where general logs are stored, including training performance and experiment details.
- **`exp_train/`**: Directory to save experiment outputs, such as model checkpoints and evaluation results.
- **`log_tft/`**: Directory dedicated to TensorFlow logs during training. This is useful for tracking the training process and debugging with TensorBoard.
- **`weights/`**: Directory to store weight of model if you want.

## Getting Started

### Prerequisites
To run this project, you need:
- **Python 3.11+**
- All required libraries listed in `requirements.txt`
- If you're using EC2 or a SageMaker Notebook, simply select the TensorFlow GPU AMI or the TensorFlow kernel, but with EC2 and use script "train.py" instead of the Jupyter Notebook, please use pip to install python-dotenv 
```bash
pip install python-dotenv==1.0.1
```

Install dependencies by running:
```bash
pip install -r requirements.txt
```

### Setting Up Environment Variables for EC2 connect to S3 database via Boto3

To configure access to AWS services, you need to set up environment variables for your AWS credentials. This project uses a `.env` file to securely store these credentials.

##### Steps to Set Up

1. **Create a `.env` file** in the root directory of your project if it doesn’t already exist.
2. **Add the following variables** to the `.env` file:

    ```plaintext
    AWS_ACCESS_KEY_ID=""
    AWS_SECRET_ACCESS_KEY=""
    AWS_SESSION_TOKEN=""
    ```

    - Replace `""` with your actual AWS credentials.
    - **AWS_ACCESS_KEY_ID**: Your AWS Access Key ID.
    - **AWS_SECRET_ACCESS_KEY**: Your AWS Secret Access Key.
    - **AWS_SESSION_TOKEN**: (Optional) AWS Session Token if required.

3. **Save the file** after entering your credentials.

#### Important Notes

- **Do not share** your `.env` file or commit it to version control as it contains sensitive information.
- For added security, consider using a `.gitignore` file to exclude `.env` from being tracked.

#### Example

An example of what your `.env` file should look like:

```plaintext
AWS_ACCESS_KEY_ID="your-access-key-id"
AWS_SECRET_ACCESS_KEY="your-secret-access-key"
AWS_SESSION_TOKEN="your-session-token"  # Optional
```


## Usage
## Dataset
The dataset used for this project is available on Kaggle. You can download it directly from the link below:
- [Kaggle Dataset - Facial-Deocclusion with face Mask](https://www.kaggle.com/datasets/ducbvh/facial-deocclusion-with-face-mask)
Please download the dataset and place it on [S3 Bucket].

- The dataset is organized into `train`, `validation`, and `test` folders. Each of these folders contains two subfolders:

    - **Mask**: Contains images with masks applied, named in the format `[image_index]_Mask.jpg`.
    - **Unmask**: Contains the original images from CelebA, named in the format `[image_index].png`.

Each pair of images in the `Mask` and `Unmask` folders shares the same `image_index`, allowing for easy mapping between masked and unmasked versions.

### Example Structure
```plaintext
dataset/
├── train/
│   ├── Mask/
│   │   ├── 0001_Mask.jpg
│   │   ├── 0002_Mask.jpg
│   │   └── ...
│   └── Unmask/
│       ├── 0001.png
│       ├── 0002.png
│       └── ...
├── validation/
│   ├── Mask/
│   └── Unmask/
└── test/
    ├── Mask/
    └── Unmask/
---
```

Make sure you have the necessary permissions and Kaggle API setup if you are downloading it programmatically.

## Train Model with JupyterNotebook
After uploading the dataset to S3, you can create a notebook instance to train your model with Tensorflow kernel. Detailed instructions are provided in this notebook file [Face_de_occlusion.ipynb](Face_de_occulusion.ipynb)


## Train Model on EC2 with GPU

If you want to train the model on an EC2 instance with GPU for easier and faster experimentation, follow these steps:

### Preparation

1. **Launch an EC2 Instance with GPU**:
   - Select an instance type with GPU support. For example, I used the **g5.xlarge** instance type, which provides a balance of GPU power and affordability.
   - Choose an AMI that includes TensorFlow GPU support, such as the **AWS Deep Learning AMI** (Amazon Machine Image) with TensorFlow pre-configured.

2. **Set Up Your Environment**:
   - Once your instance is running, ensure you have the necessary dependencies installed. The required packages can be found in the `requirements.txt` file:
     ```bash
     pip install -r requirements.txt
     ```

    - If you launch EC2 instance, selected AMI Tensorflow, you  may not install all package in requirement, only install python-dotenv 
        ```bash
        pip install python-dotenv==1.0.1
        ```
   
3. **Run Training Script**:
   - Use the `train.py` script to start training your model on the GPU instance:
     ```bash
     python train.py
     ```
### Training Script Arguments

The `train.py` script includes several configurable arguments that allow you to customize the training process. Below is a list of the available arguments:

#### Arguments

| Argument            | Type    | Default                           | Description |
|---------------------|---------|-----------------------------------|-------------|
| `--s3_bucket`       | `str`   | `"aws-ml-mycapstone-project"`     | Name of the S3 bucket to store or retrieve data. |
| `--data`            | `str`   | `"dataset-try"`                   | Name of the dataset used for training. |
| `--lambda_rec`      | `float` | `1.2`                             | Value for the reconstruction loss. |
| `--lambda_adv`      | `float` | `0.5`                             | Value for the adversarial loss. |
| `--lambda_ssim`     | `float` | `80`                              | Value for the SSIM loss. |
| `--batch_size`      | `float` | `8`                               | Batch size for training. |
| `--num_epochs`      | `int`   | `210`                             | Total number of epochs for training. |
| `--epoch_updateD`   | `int`   | `15`                              | Interval in epochs to update discriminator. |
| `--learning_rate`   | `float` | `0.5`                             | Learning rate for the optimizer. |
| `--model_dir`       | `str`   | `"./weights/generator_base.h5"`   | Directory to save the model weights. |
| `--output_dir`      | `str`   | `"./exp_train"`                   | Directory to store training results and logs. |
| `--epoch_save_results` | `int` | `5`                              | Interval in epochs to save training results. |
| `--num_img_trains`  | `int`   | `0`                               | Number of images to use for training (useful for limiting dataset size in testing). |
| `--load_retrain`    | `bool`  | `False`                           | Whether to load model weights for retraining. |
| `--save_cp_epoch`   | `bool`  | `True`                            | Whether to save checkpoints at each epoch. |
| `--train_attention` | `bool`  | `False`                           | Whether to train the model with an attention module. |

### Example Usage

To run `train.py` with custom arguments, use the following command:

```bash
python train.py --s3_bucket "your-s3-bucket-name" --data "custom-dataset" --num_epochs 100 --learning_rate 0.001
```

### Inference/Test the Model

After training, you can test the model using the `inference.py` script. This script allows you to load a trained model and run inference on new images.

#### Arguments for `inference.py`

The `inference.py` script supports the following arguments:

| Argument      | Type    | Default                         | Description |
|---------------|---------|---------------------------------|-------------|
| `--model_dir` | `str`   | `"./weights/generator_base.h5"` | Path to the trained model file. |
| `--output_dir`| `str`   | `"./inference_output"`          | Directory where the inference output will be saved. |
| `--image_path`| `bool`  | `False`                         | Path to the image(s) for inference. Set this to `True` to specify an image path. |

#### Example Usage

To run `inference.py` with custom arguments, use the following command:

```bash
python inference.py --model_dir "./path/to/your/model.h5" --output_dir "./path/to/output" --image_path "./path/to/image.jpg"
```

By using a GPU-enabled EC2 instance, you'll benefit from faster model training and the flexibility to run more complex models without as much latency. This setup is ideal for deep learning experimentation and large-scale model training.