# 👁️‍🗨️ Diabetic Retinopathy Detection App

A fast, themed diagnostic app that predicts diabetic retinopathy from retina images using a trained **SVM model**.  
Built with **Streamlit**, the app enables users to upload retina scans and receive instant DR stage predictions.

> 😈 Project styled with a dark, devilish theme — made by **King**

---

## 🧠 Model

- **Type**: Support Vector Machine (SVM)
- **Input**: Preprocessed fundus images
- **Output**: One of five stages of Diabetic Retinopathy (0–4)
- **Model size**: ~235 MB (**excluded** from GitHub repo)

📦 [Download model file](https://drive.google.com/file/d/1ULij3MiZPSWJ-EZaClFBtHZAmjOL_eIk/view?usp=sharing)  
⬇️ Save `svm_model.pkl` in the project root before running the app.

---

## 🧪 DR Stage Labels

| Code | Diagnosis           | Risk Level                     |
|------|---------------------|--------------------------------|
| 0    | No DR               | ✅ Low risk                     |
| 1    | Mild DR             | ⚠️ Mild risk                    |
| 2    | Moderate DR         | ⚠️ Moderate risk                |
| 3    | Severe DR           | ❗ High risk – needs attention  |
| 4    | Proliferative DR    | 🚨 Critical – urgent attention  |

---

## ⚙️ Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/kiran2918/Diabetic_Retinopathy.git
   cd Diabetic_Retinopathy
💻 Features
Upload retina scan images (.jpg, .png)

Real-time classification using SVM

Histogram Equalization + Grayscale preprocessing

Fully interactive UI built on Streamlit

Devil-inspired theme 🔥

📸 Screenshots (Optional)
You can add screenshots like:

assets/upload.png (upload page)

assets/result.png (prediction result)

🛡️ Disclaimer
This app is for educational and research purposes only.
It is not a replacement for medical advice or diagnosis.
Always consult a licensed medical professional for clinical decisions.

🧾 Tech Stack
Python
Streamlit
Scikit-learn
OpenCV
scikit-image
PIL (Pillow)

😈 Footer
Built with 🔥 darkness & code by King
Guided by shadows. Powered by vision.
