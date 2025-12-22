# 🎓 Semester-Project-III

## 🖼️ Advanced Image Captioning & Vision Analytics System

An end-to-end deep learning system that automatically generates natural language captions for images, performs visual analytics, benchmarks performance, and exports professional PDF reports.

==================================================

## 🚀 Features

- Upload any image → auto-generate caption
- CNN + LSTM model (InceptionV3 + LSTM)
- Tokenizer persistence for inference
- Object detection & heatmaps
- Visual analytics & graphs
- Performance benchmarking (CPU / GPU / Memory / Latency)
- PDF report generation
- REST API (FastAPI)
- Google Colab ready

==================================================

## 📂 Project Structure

.
├── caption_model.keras
├── caption_model.h5
├── tokenizer_for_training.pkl
├── inference.py
├── api_server.py
├── requirements.txt
├── model_report_outputs/
└── README.txt

==================================================

## 🛠 Tech Stack

- TensorFlow / Keras
- PyTorch / TorchVision
- NumPy, Matplotlib
- FastAPI
- Google Colab

==================================================

## 1️⃣ Train / Build and Save Model

Run the provided notebook cell that:

- Uploads an image
- Extracts features using InceptionV3
- Builds the CNN + LSTM caption model

Saves:

caption_model.keras
caption_model.h5
tokenizer_for_training.pkl

Files will auto-download in Google Colab.

==================================================

## 2️⃣ Load Model & Generate Captions

Use the inference script:

python inference.py path/to/image.jpg

==================================================

## 3️⃣ Run Full Vision Pipeline

Upload images → the system will:

- Generate captions
- Detect objects
- Draw bounding boxes
- Create heatmaps
- Plot graphs
- Benchmark performance
- Generate a PDF report

All outputs are saved in:

model_report_outputs/

==================================================

## 📊 Visualizations Generated

- Object distribution bar chart
- Detection confidence histogram
- Bounding box size violin plot
- Attention heatmap overlay
- Object co-occurrence matrix
- Caption word frequency plot

==================================================

## ⏱️ Performance & Benchmarking

The system automatically records:

- Caption inference time
- Detection inference time
- End-to-end latency
- CPU utilization (%)
- GPU utilization (%)
- Memory usage (%)

Graphs Included:
- Inference time histogram
- CPU/GPU utilization over time
- Memory profile during batch processing

==================================================

## 📄 PDF Report

Each run generates a professional report containing:

- Metadata summary
- System configuration
- Generated captions
- Detection overlays
- Key analytics plots
- Benchmark summaries

Saved as:
model_report_YYYYMMDDTHHMMSSZ.pdf

## Link : https://1b1a4e3fe5b151f737.gradio.live/

==================================================

## 🧪 Example Output

📌 Caption:
A family playing in the grass with a hose.

📌 Detections:
- person (0.98)
- dog (0.87)
- hose (0.76)

==================================================

## ❗ Error Handling

The system includes:

- Safe model loading
- Try/except on every inference step
- Graceful fallbacks if GPU or models are unavailable
- Informative logs instead of crashes

==================================================

## 📜 License

This project is licensed under the MIT License.
You are free to use, modify, and distribute for research and commercial use.

==================================================

## 🙌 Acknowledgements

- TensorFlow & Keras Team
- PyTorch & TorchVision
- HuggingFace Transformers
- Google Colab
- MSCOCO Dataset

==================================================

⭐ If you find this project useful
Please star the repository and share it with others!

==================================================

Author: Saurabh Badgujar

Email: saurabhbadgujar851@gmail.com

GitHub: https://github.com/saurabh-001-rb
