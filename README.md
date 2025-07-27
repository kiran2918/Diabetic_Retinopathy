##🔬 InfernoSight: Diabetic Retinopathy Detection
Detect diabetic retinopathy severity from retina images using a handcrafted image processing pipeline and an SVM classifier.

<!-- optional if you add a visual -->

🚀 Demo
🔗 Try it live on Hugging Face Spaces
📦 Model: SVM trained on processed retinal images
🖼️ Input: JPG, PNG, or JPEG retina scans
📊 Output: DR severity class (0 to 4) + risk level

📂 Project Structure
bash
Copy
Edit
Diabetic_Detection/
│
├── app.py                 # Streamlit app
├── main.py                # Training and pipeline script:(coming soon)
├── svm_model.pkl          # Trained SVM model
├── data/
│   └── train.csv          # Training metadata
├── notebook.ipynb         # EDA and pipeline dev notebook
├── .gitignore
└── README.md              # Project documentation
🧠 DR Stages Explained
Class	Stage	Risk Level
0	No DR	✅ Low risk
1	Mild	⚠️ Mild risk
2	Moderate	⚠️ Moderate risk
3	Severe	❗ High risk – needs attention
4	Proliferative DR	🚨 Critical – urgent care needed

🧰 Tech Stack
Python 3.10+

OpenCV, NumPy, scikit-image

scikit-learn, joblib

Streamlit for web interface

⚙️ Features
🧠 Image pre-processing pipeline: grayscale → resize → histogram equalization

🧪 SVM model trained on structured retinal image features

🎯 Accurate risk prediction with intuitive UI

💡 Fully reproducible with .ipynb and main.py training script

📦 Installation & Run
Clone the repo

bash
Copy
Edit
git clone https://github.com/kiran2918/diabetic_Retinopathy.git
cd diabetic_Retinopathy
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the app

bash
Copy
Edit
streamlit run app.py
🌐 Screenshots
Upload	Prediction

(Add screenshots in a docs/ folder or remove this section)

👨‍⚕️ Use Case
For early screening of diabetic retinopathy in telemedicine setups

Educational and prototyping use in healthcare AI

🙋‍♂️ Author
👑 Built by King
🧠 Fueled by vision, powered by SVM
😈 Summoned from the shadows — just like DR sneaks in

⚠️ Disclaimer
This is a proof-of-concept educational project. It is not approved for medical diagnosis or clinical use.
