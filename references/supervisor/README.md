# 💊 Medicine Recommendation System: AI-Powered Healthcare Assistant 🤖🩺

<p align="center">
  <img src="https://github.com/dr-mushtaq/Medicine-Recommendation-System/blob/main/390d268e-5a56-4d73-a56e-57fc740d2d85.png" width="800"/>
</p>

---

## 🎯 Overview

The **Medicine Recommendation System** is an advanced AI-powered virtual assistant designed to enhance patient care by providing **accurate and personalized medicine suggestions**.

It uses **Machine Learning (ML)** and **Natural Language Processing (NLP)** to analyze symptoms and recommend possible treatments.

⚠️ **Disclaimer:** This project is for **educational and research purposes only** and does NOT replace professional medical advice.

---

## 🎯 Objectives

- Implement **NLP-based chatbot interaction**
- Provide **personalized medicine recommendations**
- Support **multi-platform deployment (web/mobile)**
- Enable **real-time analytics**
- Build a **scalable and modular system**

---

## 🏗️ System Architecture

### 1. AI & Recommendation Engine
- Symptom analysis
- Disease prediction
- Medicine recommendation
- Models: Transform Base models

### 2. Chatbot Interface
- User-friendly UI (Streamlit / Flask)
- Accepts natural language input
- Real-time responses

### 3. Data & Analytics
- Stores user queries
- Tracks performance
- Improves recommendations
## 📌 Key Features

### 💡 AI-Powered Recommendations
- Suggests medicines based on symptoms  
- Personalized outputs using user data  
- Context-aware responses  

### 💬 Conversational Chatbot
- Natural language interaction  
- Real-time query handling  
- Symptom-based diagnosis assistance  

### 📊 Data-Driven Insights
- Tracks user behavior  
- Improves model accuracy over time  
- Provides analytics dashboard  

### 🌐 Multi-Platform Support
- Web application  
- Mobile-friendly interface  
- API integration support  

### ⚙️ Scalable & Modular
- Easy to extend with new diseases or datasets  
- Supports integration with healthcare systems  

---
## 🧪 Example Interaction

**User Input:**

I have fever, headache, and body pain


**System Output:**

Possible Condition: Viral Fever
Recommended Medicine: Paracetamol

"⚠️ This may indicate Viral Fever. Commonly used medication includes Paracetamol, but please consult a doctor before taking any medicine."

## 🧰 Technology Stack

| Component | Tools |
|----------|------|
| **ML Models** | Scikit-learn, XGBoost, TensorFlow |
| **NLP** | NLTK, SpaCy, Transformers (BERT) |
| **Backend** | Flask / FastAPI |
| **Frontend** | Streamlit / HTML / React |
| **Database** | MySQL / Firebase |
| **Deployment** | AWS / Heroku / Docker |

---
Domain-Specific Transformer Models for Medical NLP
| Model               | Training Data                              | Strengths                                                                | Best Use Case                                                              | Notes                                       |
| ------------------- | ------------------------------------------ | ------------------------------------------------------------------------ | -------------------------------------------------------------------------- | ------------------------------------------- |
| BioBERT             | PubMed abstracts + biomedical text corpora | Strong biomedical concept extraction and relation understanding          | Medical literature mining, symptom/diagnosis extraction from research text | General biomedical NLP benchmark model      |
| ClinicalBERT        | Clinical notes (e.g., MIMIC-III dataset)   | Better at real-world clinical language, abbreviations, and patient notes | Hospital records, symptom descriptions, EHR analysis                       | Often outperforms BioBERT on clinical tasks |
| Clinical-Longformer | Clinical datasets + long-sequence training | Handles long documents (up to ~4096 tokens) efficiently                  | Long patient histories, discharge summaries, full medical reports          | Designed for long-context medical NLP       |


## 🚀 Development Plan

### Phase 1: Prototype
- Train ML model on medical dataset  
- Build basic chatbot  
- Test symptom prediction


-
### Phase 2: 🧪 Evaluation
- Accuracy alone is not enough in healthcare.
- Confusion matrix  
- ROC-AUC
- False negative rate (VERY important in medical apps)  

### Phase 3: Enhancement
- Improve recommendation accuracy  
- Add user history tracking  
- Integrate mobile/web interface  

### Phase 4: Deployment
- Deploy full system  
- Add multilingual support  
- Enable real-time analytics  

## 🔮 Future Scope
- Integration with hospital systems  
- AI-based disease prediction  
- Voice-enabled medical assistant  
- Wearable device integration  
- Personalized healthcare dashboards  

---

## 📬 Contact

For queries or collaboration:

- 📧 Email: mushtaqmsit@gmail.com  
- 💼 LinkedIn: https://www.linkedin.com/in/mushtaq-hussain-21417814/  
- 🎥 YouTube: https://www.youtube.com/@coursesteach-mv5si
-  Team ID: themushtaq48

---

## ⭐ Support

If you find this project useful, please ⭐ star the repository and share it!

---

## ⚠️ Important Note

This system:

- ❌ Does NOT replace doctors  
- ❌ Should NOT be used for real medical decisions  
- ✅ Intended for learning, research, and prototyping only  

---

## 🚀 Let’s Build Smarter Healthcare with AI!


Star this repo if you find it useful ⭐

<details> 
<summary> <h2>📚Chapter: 1  -Literature Review</h2> </summary>

## 📚Section: 1  - **Samman Arooj**
 
|Title| Published Date| Research Questions| Model performance metrics|Research Gap|Taking Notes|Input/Target Features|Journal name/ Category|Limitations|Future Directions| 
|---|---|---|---|---|---|---|---|---|---|
|**🌐1- The development of a chatbot using Convolutional Neural Networks**| 2022 |1.How do different CNN architectures impact chatbot performance?|1.Accuracy      2.Training Speed|Did author compared  CNN chatbot to simpler models or rule-based systems? How well does it handle complex or open ended questions?| |
|**🌐2- Machine learning algorithms for teaching AI chat bots** | 2021 |1. Which machine learning algorithms are most successful in training AI chatbots for various tasks?|N/A|The paper does not cover various methods for evaluating the effectiveness of chatbot training algorithms. How can we measure a chatbot's ability to hold natural conversations, understand user intent, and generate appropriate responses?|Microservice architecture is used and the speed of message processing and preparation of responses by the chatbot will not change depending on the load on the server and the number of incoming messages. |
|**🌐3- Personified Robotic Chatbot Based On Compositional Dialogues** | 2022 |Research likely doesn't focus on specific questions but rather explores how compositional dialogues (where conversations are built from smaller elements) can be used to create a personified robotic chatbot.|N/A|1. How effectively can the level of personality be measured in these chatbots?          2.Is user perception the only metric, or can objective measures be developed?| |
| **🌐4- Boosting the Accuracy of Optimization Chatbot by Random Forest with Halving Grid Search Hyperparameter Tuning**| 2023 |1.Can hyperparameter tuning with a Halving Grid Search method improve the accuracy of an optimization chatbot built using a Random Forest algorithm?|1.Accuracy    2.Precision    3.Recall  |The paper proposes three chatbot models: 1.One without hyperparameter tuning 2.One with hyperparameter tuning using Halving Grid Search 3.One with hyperparameter tuning and the best performing settings| |   
|**🌐5- Developing a Chatbot using Machine Learning**| 2019|1.Can machine learning algorithms improve the ability of a chatbot to understand natural language queries?  2. How does the choice of machine learning model (e.g., recurrent neural networks, decision trees) impact the performance of a chatbot?|1. BLEU Score (BiLingual Evaluation Understudy)  2.Turing Test|This paper does not Investigate the impact of different visual design elements on user attention and engagement with the chatbot.| |
|**🌐6- Designing a Chatbot for Contemporary Education: A Systematic Literature Review**| 2023|What are the steps for designing an educational chatbot for contemporary education?|N/A|It focuses on the development of chatbots for education, not the impact on learners or educators.| |
|**🌐7- Research on the Design of Intelligent Chatbot Based on Deep Learning** | 2021 |It  is likely centered around improving the response generation of chatbots built with deep learning techniques.|N/A|Research gaps could exist in areas like sentiment analysis and generation of emotionally responsive dialogue.|Paper proposes an improved two-way GRU + Attention model based on the idea of mutual information, and examines the quality of the model from the final response effect. |
|**🌐8- Question Answering Model Based Conversational Chatbot using BERT Model and Google Dialogflow** | 2021 |N/A|N/A|The focus might be on question answering. Future research could explore integrating functionalities like sentiment analysis to tailor responses to user emotions or incorporating functionalities for completing tasks beyond just answering questions.|The focus of the paper seems to be on building and demonstrating the feasibility of a question-answering chatbot using BERT and Dialogflow. It describes the architecture and functionalities of the chatbot|
|**🌐9- Chatbot : A Question Answering System for Student** | 2021|It suggests the research question that  revolves around developing a chatbot system that effectively functions as a question answering system for students.|N/A|N/A|Paper discusses the design and development of such a chatbot, including the challenges of creating a system that can understand and answer student queries effectively.|
|**🌐10-QAM: Question Answering System Based on Knowledge Graph in the Military** | 2020|How can a knowledge graph-based Question Answering System (QAM) be effectively designed to be used in the military domain?|N/A|Slice of words not included in the JIEBA will be divide, which cause that the following steps can’t accuracy judged. And some unclear words often led to the system failed to judge the right answer and return a wrong answer to the user.|Research used the tool of NEO4J to build the military KG as well python to construct QA system|
</details>

<details> 
<summary> <h2>📚Chapter: 02 -Model metrics benchmarks </h2> </summary>

   ##  📚Chapter: 3  **Model metrics benchmarks**

|   | f1  | precision  |  recall |  accuracy | training_time  |  inference_time |
|---|---|---|---|---|---|---|
| NB	| 0.830188679245283	| 0.8461538461538461| 0.8148148148148148| 0.8085106382978723	| 0.0032272338867187	| 0.0007801055908203|
| LR	| 0.8518518518518519	| 0.8846153846153846	| 0.8214285714285714| 0.8297872340425532| 0.0356471538543701| 0.0002150535583496|
| KNN	| 0.7058823529411765| 0.6923076923076923| 0.72|	0.6808510638297872|	0.0005688667297363	| 0.0972669124603271|
| SVM	| 0.8518518518518519	| 0.8846153846153846	| 0.8214285714285714	| 0.8297872340425532| 0.0274648666381835	| 0.0030479431152343|
| XGBoost	| 0.9122807017543859	| 1.0	| 0.8387096774193549	| 0.8936170212765957| 0.241973876953125| 0.0040738582611083984|
| RoBERTa| 0.9230769230769231| 0.9230769230769231| 0.9230769230769231	| 0.9148936170212766| 24968.250607967377| 68.44634509086609|
</details>


<details> 
<summary> <h2>📚Chapter: 3 - 📚Implementation </h2> </summary>
 
  ## 📚Chapter:3.1-**Model Training**
  
| Title | Code | Dataset |
|---|---|---|
|[**2021-Selection of the Right Undergraduate Major by Students Using Supervised Learning Techniques**](https://www.mdpi.com/2076-3417/11/22/10639)|[![Colab icon](https://img.shields.io/badge/Colab-Open-blue.svg?logo=colab&logoColor=white)](https://github.com/hussain0048/Research-Papers/blob/main/Selection_of_the_Right_Undergraduate_Major_.ipynb)|[Dataset](https://github.com/hussain0048/Research-Papers/blob/main/Placement_Data_Full_Class.csv)|---|
| [**2022-Machine Learning-Driven Approach for a COVID-19 Warning System**](https://www.mdpi.com/2079-9292/11/23/3875) | [![Colab icon](https://img.shields.io/badge/Colab-Open-blue.svg?logo=colab&logoColor=white)](https://github.com/hussain0048/Research-Papers/blob/main/5_13_2020_Covid_LSTM.ipynb) [![Colab icon](https://img.shields.io/badge/Colab-Open-blue.svg?logo=colab&logoColor=white)](https://github.com/hussain0048/Research-Papers/blob/main/AR_and_ARIMA_Models.ipynb) | [Dataset](https://github.com/hussain0048/Research-Papers/blob/main/data_cases1.csv) |---|

## 📚Chapter:3.2- **Apps Details**
 
|Title| Public_URL| Deployed Repository link| Tools Details|Notbook|Dataset|Medium|Diagram|
|---|---|---|---|---|---|---|---|
|[**🌐1- Thesis**](https://medium.com/@Coursesteach/machine-learning-part-1-31bdf37404ee) | [1](https://drive.google.com/file/d/1JyDUmJ9U6mUlCvwBvC6crxVpdxbup9iH/view?usp=sharing)[-2](https://www.youtube.com/watch?v=sVsF_Ne_J6c&list=PLRKtJ4IpxJpDxl0NTvNYQWKCYzHNuy2xG&index=10)[-2]


## 📚Chapter:3.3-**Tools and Techniques Details**
| Topic Name/Tutorial | Video | Code |
|---|---|---|
| [**🌐-1-Preprocessing in Machine Learning**](https://medium.com/@Coursesteach/guide-to-supervised-learning-with-scikit-learn-part-4-501068cf021) | [1](https://drive.google.com/file/d/14MyKUWqykavcOp2MNIgQjGVU1TOyqwGg/view) [-2](https://drive.google.com/file/d/19Sx937C_K5JWQYvdv7h2J2aRdiHiucAS/view?usp=sharing)| |
|[**🌐2- Importing the Data Set Using Scikit-Learn**](https://medium.com/@Coursesteach/guide-to-supervised-learning-with-scikit-learn-part-6-importing-the-dataset-6b7e133fca66)|---|[![Colab icon](https://img.shields.io/badge/Colab-Open-blue.svg?logo=colab&logoColor=white)](https://github.com/hussain0048/Machine-Learning/blob/master/Data_Processing_in_Python_.ipynb)|
|[**🌐3-Handling missing data**](https://medium.com/@Coursesteach/supervised-learning-with-scikit-learn-part-7-handling-missing-data-b1b6263ce996)|[1](https://drive.google.com/file/d/1dN_YRnwuUf8QpUWeSnLEqHm-PtIWoPuF/view)|[![Colab icon](https://img.shields.io/badge/Colab-Open-blue.svg?logo=colab&logoColor=white)](https://github.com/hussain0048/Machine-Learning/blob/master/Data_Processing_in_Python_.ipynb)|
|[**🌐4-Data Imbalanced problem**](https://medium.com/@Coursesteach/supervised-learning-with-scikit-learn-part-8-data-imbalanced-problem-9e307c368a4d)|[1](https://drive.google.com/file/d/1Dcu0uZfT_zFmPrMUS1DkeDNKgA83Nodt/view?usp=sharing)|[![Colab icon](https://img.shields.io/badge/Colab-Open-blue.svg?logo=colab&logoColor=white)](https://github.com/hussain0048/Machine-Learning/blob/master/Data_Processing_in_Python_.ipynb)|
|[**🌐5-Data Transformation**](https://medium.com/@Coursesteach/supervised-learning-with-scikit-learn-part-9-data-transformation-b83ba14b1a2d)|[1](https://drive.google.com/file/d/14MyKUWqykavcOp2MNIgQjGVU1TOyqwGg/view)[-2](https://drive.google.com/file/d/1uY6x3O2G2f_jhngzdjUrXmwVB1o5QhEO/view?usp=sharing)|[![Colab icon](https://img.shields.io/badge/Colab-Open-blue.svg?logo=colab&logoColor=white)](https://github.com/hussain0048/Machine-Learning/blob/master/Data_Processing_in_Python_.ipynb)|
|[**🌐4-Centering and scaling.**](https://medium.com/@Coursesteach/supervised-learning-with-scikit-learn-part-10-centering-and-scaling-08c914162f81)|[1](https://drive.google.com/file/d/1gG742Q_qVbDuRbPMzJjGT_Hx1d-Joz4j/view)[-2](https://drive.google.com/file/d/1ivw7tVzaiecaJRpzoei6azhBDQySySbJ/view?usp=sharing)|[![Colab icon](https://img.shields.io/badge/Colab-Open-blue.svg?logo=colab&logoColor=white)](https://github.com/hussain0048/Machine-Learning/blob/master/Data_Processing_in_Python_.ipynb)|
|[**🌐5-Removing Outliers**](https://medium.com/@Coursesteach/supervised-learning-with-scikit-learn-part-11-removing-outliers-with-scikit-learn-59d6a2051d02)|[1](https://drive.google.com/file/d/1NhUQQx0e2s-oG6oLJKpgNxn-_NBXRQVl/view?usp=sharing)[-2](https://drive.google.com/file/d/1DdFCHKbJm8LU5mqDE9J6PbaepGxGUglY/view?usp=sharing)|[![Colab icon](https://img.shields.io/badge/Colab-Open-blue.svg?logo=colab&logoColor=white)](https://github.com/hussain0048/Machine-Learning/blob/master/Data_Processing_in_Python_.ipynb)|
|[**🌐6-Data Splitting**](https://medium.com/@Coursesteach/supervised-learning-with-scikit-learn-part-12-data-splitting-07658730bb01)|[1](https://drive.google.com/file/d/1vpTQiPWqO-_kb18Tt3L01ZMamFEAG6eT/view)[-2](https://www.youtube.com/watch?v=6dbrR-WymjI&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A&index=8)[-3](https://drive.google.com/file/d/1nhSSLP2bejY_49r-5m2yCZuaTyv2FO38/view)[-4](https://www.youtube.com/watch?v=ivVeqv4oShk&list=PLTKMiZHVd_2KyGirGEvKlniaWeLOHhUF3&index=61)|[![Colab icon](https://img.shields.io/badge/Colab-Open-blue.svg?logo=colab&logoColor=white)](https://github.com/hussain0048/Machine-Learning/blob/master/Data_Processing_in_Python_.ipynb)|
|[**🌐7-Pipelines in scikit-learn**](https://medium.com/@Coursesteach/supervised-learning-with-scikit-learn-part-14-pipelines-in-scikit-learn-dc408eb152d1)|[1](https://drive.google.com/file/d/1A00SABP9KsLGwET-sSA03G7M3byA2j8u/view)[-2](https://www.youtube.com/watch?v=MuPmbW0ln6g&list=PLTKMiZHVd_2KyGirGEvKlniaWeLOHhUF3&index=31)|[![Colab icon](https://img.shields.io/badge/Colab-Open-blue.svg?logo=colab&logoColor=white)](https://github.com/hussain0048/Machine-Learning/blob/master/Pipelines_in_scikit_learn.ipynb)|

## 📚Chapter:4-**Measuring model performance**
| Topic Name/Tutorial | Video | Code |
|---|---|---|
| [**🌐-1-Introduction of Model Evaluation**](https://medium.com/@Coursesteach/supervised-learning-with-scikit-learn-part-15-introduction-of-model-evaluation-3906e2b1bb1a) |---|--- |
|[**🌐2- Confusion Metrix**](https://medium.com/@Coursesteach/supervised-learning-with-scikit-learn-part-16-confusion-metric-8c0dad493d92)|---|[![Colab icon](https://img.shields.io/badge/Colab-Open-blue.svg?logo=colab&logoColor=white)](https://github.com/hussain0048/Machine-Learning/blob/master/Data_Processing_in_Python_.ipynb)|
|[**🌐3-Handling missing data**](https://medium.com/@Coursesteach/supervised-learning-with-scikit-learn-part-7-handling-missing-data-b1b6263ce996)|[1](https://drive.google.com/file/d/1dN_YRnwuUf8QpUWeSnLEqHm-PtIWoPuF/view)|[![Colab icon](https://img.shields.io/badge/Colab-Open-blue.svg?logo=colab&logoColor=white)](https://github.com/hussain0048/Machine-Learning/blob/master/Data_Processing_in_Python_.ipynb)|
</details>


<details> 
<summary> <h2>📚Chapter:5-Resources </h2> </summary>
  
 - [**Personalized-Medical-Recommendation-System-with-Machine-Learning | Generative AI | LLM**](https://github.com/611noorsaeed/Medicine-Recommendation-System-Personalized-Medical-Recommendation-System-with-Machine-Learning)
 - [Personalized-Medical-Recommendation-System-with-Machine-Learning](https://github.com/611noorsaeed/Medicine-Recommendation-System-Personalized-Medical-Recommendation-System-with-Machine-Learning/tree/main)
 - [ContextFlow-Recommender](https://github.com/Rafff-ml/ContextFlow-Recommender)
 - [Medicine Recommendation System Dataset](https://www.kaggle.com/datasets/noorsaeed/medicine-recommendation-system-dataset?resource=download)
 - [Build & Deploy ML Regression model with FastAPI, MLFlow, Docker, & AWS](https://youtu.be/Y0SbCp4fUvA?si=8U9M4VG96DM9En-a)
 - [Hugging Face Hub.](https://huggingface.co/models?sort=trending)
</details>


## 💻 Workflow:

- Fork the repository

- Clone your forked repository using terminal or gitbash.

- Make changes to the cloned repository

- Add, Commit and Push

- Then in Github, in your cloned repository find the option to make a pull request 

> print("Start contributing for Machine Learning")
>
## ⚙️ Things to Note

* Make sure you do not copy codes from external sources because that work will not be considered. Plagiarism is strictly not allowed.
* You can only work on issues that have been assigned to you.
* If you want to contribute the algorithm, it's preferrable that you create a new issue before making a PR and link your PR to that issue.
* If you have modified/added code work, make sure the code compiles before submitting.
* Strictly use snake_case (underscore_separated) in your file_name and push it in correct folder.
* Do not update the **[README.md](https://github.com/prathimacode-hub/ML-ProjectKart/blob/main/README.md).**

 🔍 **Explore more👋🛒**
 
Explore cutting-edge tools and Python libraries, access insightful slides and source code, and tap into a wealth of free online courses from top universities and organizations. Connect with like-minded individuals on Reddit, Facebook, and beyond, and stay updated with our YouTube channel and GitHub repository. Don’t wait — enroll now and unleash your Machine Learning  potential!”

* [**Supervised learning with scikit-learn**](https://coursesteach.com/enrol/index.php?id=21)
* [**Fundamental of Machine Learning**](https://coursesteach.com/enrol/index.php?id=6)

<a href="https://app.commanddash.io/agent?github=<your github repo>"><img src="https://img.shields.io/badge/AI-Code%20Gen-EB9FDA"></a>

## License

This project is licensed under the MIT License - see the LICENSE file for details.

All source code and educational material in this repository are released under the MIT License

## **✨Top Contributors**
We would love your help in making this repository even better! If you know of an amazing AI course that isn't listed here, or if you have any suggestions for improvement in any course content, feel free to open an issue or submit a course contribution request.

                       Together, let's make this the best AI learning hub website! 🚀

Thanks goes to these Wonderful People. Contributions of any kind are welcome!🚀

<a href="https://github.com/hussain0048/VU-Chatbot-System/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=hussain0048/VU-Chatbot-System" />
</a>







