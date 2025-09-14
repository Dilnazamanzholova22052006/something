# 🚗 Car Condition Analyzer

Car Condition Analyzer is a [Streamlit](https://streamlit.io/) application for **automatic vehicle condition analysis** from photos.  

The system uses YOLO-based models to detect:
- **Cleanliness** → clean / dirty
- **Integrity** → damaged / not damaged
- Annotates images with bounding boxes
- Allows downloading annotated results

---

## ✨ Features
- Simple web interface (upload photo → get result)
- Dual-model pipeline (clean/dirty + damaged/not damaged)
- Annotated images with bounding boxes
- Downloadable layers (dirty/damage detection)
- Lightweight CPU support (no GPU required)

---

## 🔧 Installation & Run (without Docker)

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```
### 2. Create virtual environment
Windows (PowerShell):
python -m venv venv
venv\Scripts\activate
Linux / Mac:

python3 -m venv venv
source venv/bin/activate
### 3. Install dependencies 
pip install -r requirements.txt
### 4. Run the application
streamlit run app.py
After starting, open in your browser 👉 http://localhost:8501
⚙️ Requirements

Python 3.9 – 3.11

Dependencies from requirements.txt

YOLO model weights:

models/dirty_best.pt

models/damaged2_best.pt

(weights can be stored in the models/ folder or downloaded from Releases
).

📂 Project Structure
car-condition-analyzer/
├─ app.py                  # main Streamlit app
├─ requirements.txt        # dependencies
├─ Readme.md               # this guide
├─ models/                 # YOLO weights
│   ├─ dirty_best.pt
│   └─ damaged2_best.pt
└─ streamlit/
   └─ config.toml          # custom Streamlit theme

📊 Pipeline Architecture

User uploads a photo 🚘

YOLO Model #1 → classifies cleanliness (clean / dirty)

YOLO Model #2 → detects integrity (damaged / not damaged)

Streamlit UI displays results:

Badges & summary card

Annotated images

Download buttons

🛡 Risks & Ethics

Privacy of photos: Vehicle images may contain identifiable details (e.g., license plates, surroundings). Sensitive information should be anonymized.

Potential bias: Performance may vary depending on camera quality, lighting conditions, or geographic region, which could lead to inconsistent results.

Prototype limitations: Current model is trained on limited datasets and may fail in extreme conditions (snow, rain, unusual damages).

📌 Future Work

Data enrichment: Collect more diverse datasets (different regions, weather, cameras).

Multi-class classification: Extend beyond binary (e.g., “slightly dirty”, “heavily dirty”, “minor scratch”, “severe damage”).

Local condition adaptation: Improve robustness in snow, rain, dust, and low-light.

Product integration:

Notifications to drivers (“Your car looks dirty — consider cleaning”).

Alerts to passengers (trust & safety signals).

Signals for quality monitoring in ride-hailing platforms.

📸 Demo (example screenshots)

Main Interface


Analysis Result


📄 License

This project is intended for educational and prototyping purposes.
Please ensure compliance with data privacy laws and avoid uploading personal or sensitive data.
