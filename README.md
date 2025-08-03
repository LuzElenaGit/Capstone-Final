# CAPTSONE Project

## Effective Spam Email Classification Final Report

## Problem Statement: 
The issue of unsolicited and malicious spam emails continues to be a significant challenge to email users, leading to productivity loss, security risks, and information overload. Current email filtering mechanisms often struggle with the evolving nature of spam, resulting in both legitimate messages being incorrectly flagged as spam and malicious spam bypassing filters (false negatives). This inefficiency necessitates a more robust and adaptive approach to email classification.

This project aims to address the challenge of accurately classifying and filtering spam emails from legitimate messages through the effective utilization of machine learning algorithms. Specifically, the problem focuses on identifying and leveraging the most significant features that contribute to accurate email classification. The core problem to be solved is how to develop and evaluate a machine learning model that can reliably distinguish between "spam" and "not spam" emails.

## Model Outcomes or Predictions: 
The primary outcome of this project will be a trained machine learning model capable of generating predictions for incoming emails, categorizing them as either "spam" or "not spam." These predictions will directly inform the filtering process, aiming to automatically divert unwanted messages while ensuring the delivery of legitimate communications. The ultimate goal is to enhance user experience by providing a cleaner, more secure inbox and reducing the manual effort required to manage email.

## Data Acquisition:
The data has been acquired from Kaggle.  https://www.kaggle.com/datasets/purusinghvi/email-spam-classification-dataset.

## Data Preprocessing and Processing:
To ensure the integrity and effectiveness of the machine learning model, a rigorous data preprocessing pipeline was implemented to prepare the raw email text. This process was crucial for ensuring the data was free of inconsistencies and optimized for numerical representation.

## Data Cleaning and Consistency: 
Data cleaning was performed to standardize the text and remove irrelevant noise. Techniques employed included:
- Lowercasing: All text was converted to lowercase to treat words like "Spam" and "spam" as identical, reducing feature sparsity.
- Removing Punctuation and Numbers: Punctuation marks and numerical digits were removed as they typically do not contribute to the semantic classification of spam.
- Removing Stopwords: Common words (e.g., "the," "a," "is") that carry little discriminatory power were eliminated to focus on more meaningful terms.
- Stemming: Words were reduced to their root form (e.g., "running," "runs," "ran" become "run") to consolidate variations of the same word.
- Tokenization: The text was broken down into individual words or "tokens," forming the basic units for feature extraction.
- HTML Tag Removal: Any embedded HTML tags were stripped from the email content to ensure only the plain text was processed.
- Links Removal: URLs and hyperlinks were removed, as their presence often indicates spam but their specific content can be highly variable and less useful than other textual features.
These techniques collectively addressed inconsistencies by normalizing the text and implicitly handled "missing values" by ensuring all relevant textual information was present in a standardized format, rather than dealing with explicit nulls common in structured datasets.

## Data Splitting and Encoding: 
Following the cleaning phase, the prepared dataset was split into training and testing sets. This crucial step ensures that the model's performance is evaluated on unseen data, providing an unbiased assessment of its generalization capabilities. A standard split ratio of 80% training, 20% testing was applied to maintain representativeness.
Next step was to transform textual data into numerical features, an essential encoding step for machine learning algorithms. This step was achieved through various feature engineering techniques:
- Bag of Words (BoW): Counting the frequency of each word within an email, creating a vector representation where each dimension corresponds to a unique word in the vocabulary.
- TF-IDF (Term Frequency-Inverse Document Frequency): Assigned weights to words based on their frequency within a single email. Term Frequency across the entire dataset (Inverse Document Frequency). TF-IDF effectively highlights words that are important in a specific email but less common overall, often indicative of spam-specific language.
- Length of Email: The total character or word count of an email was included as a feature, as spam emails can sometimes exhibit characteristic lengths.
- Keyword Presence: The existence of specific keywords commonly associated with spam (e.g., "free," "win," "prize") was encoded as binary features, providing direct indicators for classification.

## Model Selection:
For this project, Neural Network and Logistic Regression models were utilized to evaluate their respective performances in spam email classification. Following are the links to the notebooks:

Logistic Regression model:
https://github.com/LuzElenaGit/Capstone-Final/blob/main/LogisticRegressionModel.ipynb

Neural Network model:
https://github.com/LuzElenaGit/Capstone-Final/blob/main/NN-Model.ipynb

Logistic Regression was selected for its efficiency in handling sparse feature vectors, a common characteristic of text data processed by methods like Bag of Words or TF-IDF. It is known for not requiring heavy computational resources or deep architectural configurations to achieve solid performance. Its speed in both training and prediction phases, coupled with its lightweight and computationally efficient nature, makes it a practical choice for real-time email filtering applications.
The Neural Network model, a feed-forward architecture, was also implemented to explore its capabilities in this classification task. 
While these two models were chosen for their specific benefits and complementary approaches, other machine learning algorithms were also considered for their potential applicability in this domain. These alternatives include:
- Naive Bayes: Particularly well-suited for text classification due to its probabilistic nature and efficiency.
- Support Vector Machine (SVM): Effective in high-dimensional spaces and capable of finding optimal separating hyperplanes.
- Random Forest / Decision Trees: Ensemble methods that can capture complex non-linear relationships within the data.
- Gradient Boosting: Another powerful ensemble technique known for its high predictive accuracy.
The selection of Neural Network and Logistic Regression represents a comprehensive approach, allowing for a comparative analysis of their performance, resource efficiency, and interpretability for the task of classifying spam emails.

## Models Evaluation:
The Logistic Regression model significantly outperforms the Neural Network in this spam classification task. It achieves an impressive overall accuracy of 97%, indicating that 97% of emails were correctly classified. Furthermore, its precision, recall, and F1-scores for both classes (0 and 1) are consistently high at 0.97, demonstrating excellent and balanced performance in identifying both legitimate and spam emails with very few false positives or false negatives.
In contrast, the Neural Network model shows a lower overall accuracy of 80%. While its precision, recall, and F1-scores are respectable (ranging from 0.74 to 0.86), they are notably lower than those of the Logistic Regression model. The Neural Network's performance is also relatively balanced between the two classes, but its lower scores across all metrics suggest it is less effective at correctly classifying emails compared to the Logistic Regression model in this specific scenario.
In summary, based on the provided metrics, Logistic Regression is the superior model for the spam email classification project, offering substantially higher accuracy and more robust performance across all evaluation metrics.
