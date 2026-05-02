## Usage Guide
The application is organised into six tabs.

*   **Tab 1: Diagnose.** Upload a clear photograph of a single crop leaf. The model returns the predicted disease class, confidence score, and a Grad-CAM heatmap. Ensure the leaf fills the frame and is well-lit for best results.
*   **Tab 2: Treatment.** Displays the full treatment advisory for the detected disease, including symptoms, chemical options, organic alternatives, preventive measures, and Hindi translation. A PDF report and WhatsApp share link are available from this tab.
*   **Tab 3: Weather Risk.** Enter a location to retrieve real-time weather conditions. The application evaluates the risk of disease spread under current conditions and provides field level advice.
*   **Tab 4: Loss Calculator.** Select the crop type and enter the cultivated area in acres. The application calculates estimated yield loss in kilograms and financial loss in INR based on the detected disease's loss percentage and current market prices.
*   **Tab 5: KVK Locator.** Select a state and enter a district to find the nearest Krishi Vigyan Kendra. A Google Maps search link is provided for each result. If a disease has been detected, the tab displays the diagnosis for the farmer to show the KVK officer.
*   **Tab 6: About.** Describes the problem context, model specifications, and full feature list.

## Model Details
The model is a custom CNN trained on the PlantVillage dataset.

*   **Input size:** 128 x 128 x 3 (RGB)
*   **Architecture:** 10 convolutional layers
*   **Parameters:** approximately 7.8 million
*   **Output:** 38 class softmax
*   **Validation accuracy:** 94.61%
*   **Explainability:** Grad-CAM is computed via TensorFlow's GradientTape on the final convolutional layer to produce a class discriminative heatmap overlaid on the original image.

## Dependencies

*   streamlit
*   tensorflow
*   numpy
*   Pillow
*   opencv-python-headless
*   requests
*   matplotlib
*   fpdf2

A complete `requirements.txt` should be generated from the project environment using `pip# AgriGuard: Crop Disease Intelligence

AgriGuard is a Streamlit based web application for artificial intelligence powered crop disease detection and agricultural advisory. It is designed to assist farmers who lack access to professional agronomists by providing instant disease diagnosis, curated treatment plans, weather based spread risk analysis, and economic loss estimates. All functionalities are available within a single interface in both English and Hindi.

## Table of Contents
* [Overview](#overview)
* [Features](#features)
* [Supported Crops and Diseases](#supported-crops-and-diseases)
* [Technology Stack](#technology-stack)
* [Project Structure](#project-structure)
* [Installation](#installation)
* [Running the Application](#running-the-application)
* [Usage Guide](#usage-guide)
* [Model Details](#model-details)
* [Dependencies](#dependencies)
* [Author](#author)

## Overview
India has over 140 million farming households. The majority of smallholder farmers do not have reliable access to agronomists, and a delayed or incorrect disease diagnosis can result in the loss of an entire season's harvest. AgriGuard addresses this gap by combining a trained convolutional neural network with a curated treatment database, real-time weather data, and a government KVK (Krishi Vigyan Kendra) locator.

## Features

### Disease Diagnosis
Upload a photograph of a crop leaf. The application runs inference using a trained CNN model and returns the predicted disease class along with a confidence score.

### Grad-CAM Explainability
A Gradient weighted Class Activation Map (Grad-CAM) overlay highlights the specific region of the leaf that the model attended to when making its prediction, providing visual justification for the diagnosis.

### Treatment Advisory
For each of the 38 supported disease classes, the application provides symptom descriptions, chemical treatment options with dosage, organic treatment alternatives, and preventive measures. All content is available in English and Hindi.

### Weather Risk Assessment
Real-time weather data is retrieved via wttr.in (no API key required). The application evaluates temperature, humidity, and precipitation to estimate the current risk of disease spread.

### Crop Loss Calculator
Given the detected disease, a user specified crop type, and the cultivated area in acres, the application estimates potential yield loss in kilograms and the corresponding economic loss in Indian Rupees (INR) at current market prices.

### PDF Report Generation
A downloadable PDF report is generated containing the diagnosis, Grad-CAM image, full treatment plan, weather assessment, and loss estimate. This report can be presented to KVK officers or agricultural extension workers.

### WhatsApp Sharing
A pre-formatted diagnostic summary can be shared via WhatsApp for remote consultation.

### KVK Locator
A searchable database of Krishi Vigyan Kendras (government farm science centres) allows users to locate the nearest centre by state and district.

## Supported Crops and Diseases
The model recognises 38 classes across 14 crops.

| Crop | Conditions |
| :--- | :--- |
| **Apple** | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| **Blueberry** | Healthy |
| **Cherry** | Powdery Mildew, Healthy |
| **Corn (Maize)** | Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| **Grape** | Black Rot, Esca (Black Measles), Leaf Blight, Healthy |
| **Orange** | Huanglongbing (Citrus Greening) |
| **Peach** | Bacterial Spot, Healthy |
| **Pepper (Bell)** | Bacterial Spot, Healthy |
| **Potato** | Early Blight, Late Blight, Healthy |
| **Raspberry** | Healthy |
| **Soybean** | Healthy |
| **Squash** | Powdery Mildew |
| **Strawberry** | Leaf Scorch, Healthy |
| **Tomato** | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |

## Technology Stack

| Component | Details |
| :--- | :--- |
| **Framework** | Streamlit |
| **Model** | Custom CNN (7.8 million parameters, 10 convolutional layers) |
| **Training Dataset** | PlantVillage (approximately 87,000 images, 38 classes) |
| **Validation Accuracy**| 94.61% |
| **Explainability** | Grad-CAM (via TensorFlow GradientTape) |
| **Weather Data** | wttr.in (no API key required) |
| **PDF Generation** | fpdf2 |
| **Image Processing** | OpenCV, Pillow |
| **Languages Supported**| English, Hindi |

## Project Structure
```text
agriguard/
|-- app.py
|-- best_model.weights.h5
|-- kvk_centers.csv
|-- requirements.txt
|-- test_model.py
|-- Test_Plant_Disease.ipynb
|-- train_and_test.py
|-- Train_plant_disease.ipynb
|-- trained_model.h5
|-- trained_model.keras
|-- training_hist.json
```

The application loads the model by attempting the following filenames in order: `trained_model.keras`, `trained_model.h5`, `trained_model`. Ensure at least one of these files is present in the project root before launching the application.

The KVK locator requires `kvk_centers.csv` in the project root. The expected columns are: state, district, address, host.

## Installation

Requirements: Python 3.9 or higher is recommended.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/agriguard.git](https://github.com/your-username/agriguard.git)
    cd agriguard
    ```

2.  **Create and activate a virtual environment** (optional but recommended):
    ```bash
    python -m venv .venv
    source .venv/bin/activate        # On Windows: .venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **File placement:**
    *   Place the trained model file (`trained_model.keras` or `trained_model.h5`) in the project root.
    *   Place `kvk_centers.csv` in the project root for the KVK Locator tab to function.

## Running the Application

Execute the following command in your terminal:
```bash
streamlit run app.py
```
The application will open in your default browser at `http://localhost:8501`.

## Usage Guide
The application is organised into six tabs.

*   **Tab 1: Diagnose.** Upload a clear photograph of a single crop leaf. The model returns the predicted disease class, confidence score, and a Grad-CAM heatmap. Ensure the leaf fills the frame and is well-lit for best results.
*   **Tab 2: Treatment.** Displays the full treatment advisory for the detected disease, including symptoms, chemical options, organic alternatives, preventive measures, and Hindi translation. A PDF report and WhatsApp share link are available from this tab.
*   **Tab 3: Weather Risk.** Enter a location to retrieve real-time weather conditions. The application evaluates the risk of disease spread under current conditions and provides field level advice.
*   **Tab 4: Loss Calculator.** Select the crop type and enter the cultivated area in acres. The application calculates estimated yield loss in kilograms and financial loss in INR based on the detected disease's loss percentage and current market prices.
*   **Tab 5: KVK Locator.** Select a state and enter a district to find the nearest Krishi Vigyan Kendra. A Google Maps search link is provided for each result. If a disease has been detected, the tab displays the diagnosis for the farmer to show the KVK officer.
*   **Tab 6: About.** Describes the problem context, model specifications, and full feature list.

## Model Details
The model is a custom CNN trained on the PlantVillage dataset.

*   **Input size:** 128 x 128 x 3 (RGB)
*   **Architecture:** 10 convolutional layers
*   **Parameters:** approximately 7.8 million
*   **Output:** 38 class softmax
*   **Validation accuracy:** 94.61%
*   **Explainability:** Grad-CAM is computed via TensorFlow's GradientTape on the final convolutional layer to produce a class discriminative heatmap overlaid on the original image.

## Dependencies

*   streamlit
*   tensorflow
*   numpy
*   Pillow
*   opencv-python-headless
*   requests
*   matplotlib
*   fpdf2

A complete `requirements.txt` should be generated from the project environment using `pip freeze > requirements.txt`.

## Author
Apoorv Aditya Tripathi
