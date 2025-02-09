<!-- PROJECT LOGO -->
<div align="center">
  <img src="/lip_reading.png" alt="Lip Reading Project Logo" width="600">
  <h3 align="center">🤖 Automated Lip Reading Using Deep Learning</h3>
  <p align="center">
   An AI-powered system that utilizes deep learning models to analyze lip movements in real-time.
  </p>
</div>
<hr>

## 📑 Table of Contents  
<details>
  <summary>▶️ Table of Contents</summary>

- [▶️ Project Overview](#-project-overview)
- [▶️ Features & Objectives](#-features--objectives)
- [▶️ Technologies Used](#-technologies-used)
- [▶️ Project Architecture](#-project-architecture)
  - [▶️ Data Collection & Preprocessing](#data-collection--preprocessing)
  - [▶️ Model Development](#model-development)
  - [▶️ Training & Evaluation](#training--evaluation)
  - [▶️ Web Application (Streamlit)](#web-application-streamlit)
- [▶️ Essential Installation Software](#-essential-installation-software)
- [▶️ Expected Output](#-expected-output)
- [▶️ Challenges Faced](#-challenges-faced)
- [▶️ Future Scope](#-future-scope)
- [▶️ Author](#-author)
</details>
<hr>

<details>
  <summary>▶️ Project Overview</summary>

The **Lip Reading Project** is an advanced AI-driven system designed to interpret spoken words solely from visual lip movements in video footage. It leverages deep learning models such as **Conv3D** for spatial feature extraction and **LSTM** for sequential analysis, enabling accurate prediction of words based on lip motion. The system processes both real-time and pre-recorded video inputs, making it a versatile tool for speech recognition without audio dependency.  

Built with **OpenCV** for video processing and **Streamlit** for an interactive web interface, this project enhances accessibility for individuals with hearing impairments, strengthens security applications, and contributes to AI-powered human-computer interaction. Its scalability and real-time processing capabilities make it a valuable innovation across various domains.  
</details>
<hr>

<details>
  <summary>▶️ Features & Objectives</summary>

- **🧠 Deep Learning Model**: Uses a combination of **3D Convolutional Neural Networks (Conv3D)** and **Long Short-Term Memory (LSTM)** networks for accurate lip movement detection.  
- **🦻 Improved Accessibility**: Aids individuals with hearing impairments by providing an alternative mode of communication.  
- **🔊 Robust Performance in Noisy Environments**: Functions effectively where traditional speech recognition systems fail.  
- **💻 Web Application Interface**: Developed using **Streamlit**, allowing users to upload video files and receive textual transcriptions.  
- **📡 Scalability**: The model can be adapted for multiple languages and real-time processing.  
</details>
<hr>

<details>
  <summary>▶️ Technologies Used</summary>

- **Programming Language**: Python  
- **Deep Learning Framework**: TensorFlow, Keras  
- **Computer Vision**: OpenCV  
- **Model Architecture**: CNN-LSTM (Conv3D + LSTM)  
- **Data Processing**: NumPy, Pandas  
- **Visualization**: Matplotlib  
- **Web Framework**: Streamlit  
</details>
<hr>

## 🏗️ Project Architecture  

<details>
  <summary>▶️ Data Collection & Preprocessing</summary>

- Downloaded video datasets with labeled speech.  
- Extracted **frames from video** and converted them into grayscale images.  
- Applied **lip detection and cropping** techniques to focus on the mouth region.  
- Normalized image data and converted it into an array for model training.  
</details>

<details>
  <summary>▶️ Model Development</summary>

- **Conv3D Layers**: Extract spatial and temporal features from video frames.  
- **MaxPooling Layers**: Reduce dimensionality for computational efficiency.  
- **LSTM Layers**: Learn sequential patterns in lip movements.  
- **Dense Layers**: Convert extracted features into text predictions.  
- **CTC Loss Function**: Used for alignment-free speech recognition.  
</details>

<details>
  <summary>▶️ Training & Evaluation</summary>

- The model was trained on a dataset of lip movements and corresponding text transcripts.  
- **Performance Metrics**: Accuracy, Precision, Recall, and WER (Word Error Rate).  
- Data split into **80% training and 20% testing** for model validation.  
</details>

<details>
  <summary>▶️ Web Application (Streamlit)</summary>

- Built an **interactive UI** where users can upload a video and receive real-time transcription.  
- Used pre-trained models to predict text from uploaded videos.  
- Displayed **frame-wise visualization** of lip movement predictions.  
</details>
<hr>

<details>
  <summary>▶️ Essential Installation Software</summary>

Ensure you have the following installed:  

- Python 3.8+  
- TensorFlow 2.x  
- OpenCV  
- Streamlit  
- NumPy  
- Pandas  
- Matplotlib  
- Imageio  
- Gdown  
</details>
<hr>

<details>
  <summary>▶️ Expected Output</summary>

- The model will display a **sequence of predicted words** corresponding to the lip movements.  
- Accuracy will depend on **lighting conditions, speaker clarity, and dataset quality**.  
</details>
<hr>

<details>
  <summary>▶️ Challenges Faced</summary>

- **📉 Dataset Limitations**: Lip-reading datasets are limited and require significant preprocessing.  
- **💻 Computational Intensity**: Training Conv3D and LSTM models requires high GPU power.  
- **👄 Speaker Variability**: Different individuals have unique lip movement styles, affecting model accuracy.  
</details>
<hr>

<details>
  <summary>▶️ Future Scope</summary>

- Implement **real-time lip reading** for live video streams.  
- Expand the dataset to support **multiple languages and accents**.  
- Optimize the model to work efficiently on **edge devices** like mobile phones.  
- Enhance the accuracy using **transformer-based architectures**.  
</details>
<hr>

<details>
  <summary>▶️ Author</summary>
<div align="center">
  <p>💡 Developed by <strong>Madhav</strong></p>
  <p>📬 Feel free to reach out for questions or contributions!</p>
  <p>🚀 Happy Coding!</p>
</div>
</details>
