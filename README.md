# Smart Image Processing & Background Extraction System

A Python-based image processing web application built using **OpenCV, NumPy, Pillow, and Streamlit**. It provides a simple interface to perform various image processing operations and extract foreground objects from images.

## 🚀 Features

- **Image Upload:** Upload images for processing.
- **Basic Processing:** Grayscale conversion, resizing, and cropping.
- **Image Enhancement:** Thresholding and histogram equalization.
- **Feature Detection:** Contour detection, blob detection, and ORB feature detection.
- **Edge and Shape Detection:** Canny edge detection and Hough line detection.
- **Image Analysis:** Image gradients, Fourier transform, and color-space conversion.
- **Feature Matching:** Match ORB features between two images.
- **Background Extraction:** Use the GrabCut algorithm to separate foreground objects from the background.
- **Download Results:** Download processed images, including transparent PNG output where supported.

## 🛠️ Technologies Used

- Python
- OpenCV
- NumPy
- Pillow
- Streamlit

## 📂 Project Structure

```text
SmartImageProcessing/
├── app.py
├── requirements.txt
├── .gitignore
├── input_images/
└── output_images/
```

## 📸 Screenshots

### 1. Main Interface
![Main Interface](screenshots/Main%20interface.png)

### 2. Image Processing
![Image Processing](screenshots/Image%20processing.png)

### 3. Background Extraction
![Background Extraction](screenshots/Background%20extraction.png)

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/aashleshagit/smart-image-processing.git
cd smart-image-processing
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The application will open in your browser, usually at `http://localhost:8501`.

## 🖥️ How to Use

1. Launch the Streamlit application.
2. Upload an image.
3. Select the image processing operation you want.
4. View the processed result.
5. Download the output image if required.

For ORB feature matching, provide the two images required by the application.

## 🎯 Project Objective

The objective of this project is to bring multiple fundamental computer vision and image processing techniques together in one easy-to-use web interface.

## 🔮 Future Enhancements

- Add more advanced segmentation techniques.
- Improve automatic foreground extraction.
- Add side-by-side comparisons of original and processed images.
- Deploy the application online for public access.

## 👩‍💻 Author

**Aashlesha Vijay Sawane**

GitHub: [@aashleshagit](https://github.com/aashleshagit)

---

*Developed as an academic mini project to explore computer vision and image processing using Python.*
