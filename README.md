# spam-detector
# 📧 Spam Email Classifier  

A **Naïve Bayes** based **Spam Email Classifier** built with **Machine Learning, NLP, and Streamlit UI**. This project allows users to input an email and determine if it's spam or not.  

---

## 🚀 Features  
✅ **Trains a Naïve Bayes Model**  
✅ **Preprocesses Emails (Removes Punctuation & Numbers)**  
✅ **Custom Dark-Themed UI with Streamlit**  
✅ **Accepts User Input and Classifies Emails**  
✅ **Saves Trained Model for Future Predictions**  
✅ **Runs Locally on `http://localhost:8501`**  

---

## 📂 Project Structure  
All functionalities (training, model saving, UI) are inside **one single `app.py` file**.  

## 🛠 Installation & Setup  

 1️⃣ Clone the Repository  
```bash
git clone https://github.com/your-username/spam-classifier.git
cd spam-classifier


#2 Create & Activate a Virtual Environment
# On Linux/Mac
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate

3️⃣ Install Dependencies
pip install streamlit scikit-learn pandas joblib

Run the App
streamlit run app.py

