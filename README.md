# Toxicity Detection in Online Comments

This project focuses on detecting and classifying toxic comments using machine learning models. The goal is to identify harmful online content efficiently and accurately, aiding content moderation and fostering healthier online interactions.

# Objective: 

The main objective of this project is to:

Detect and classify comments into three categories:
- Non-toxic
- Less toxic
- Highly toxic
Evaluate the model's performance using accuracy and confusion matrices to analyze its strengths and limitations.

**Dataset**: Utilizes a public dataset from Civil Comments, extended by Jigsaw with additional labels for toxicity and identity mentions.

**Model**: Implements the pre-trained unitary/toxic-bert model for text classification.


# Steps:
1. **Data Preparation:**

- Loaded the Civil Comments dataset from Hugging Face.
- Pre-processed the data by keeping only the text and toxicity columns.


2. **Prediction:**

Used the pre-trained "unitary/toxic-bert" model from Hugging Face to predict toxicity scores for comments.
Rounded the scores to two decimal places and labeled them based on thresholds:
- 0 : Non-toxic
- 0 to 0.5 : Less toxic
- more than 0.5 : Highly toxic

3. **Evaluation:**

Compared the model’s predictions with the actual labels.
Calculated accuracy and generated a confusion matrix to measure performance.

4. **Visualization:**

Visualized the confusion matrix using a heatmap for better understanding of misclassifications.

5. **Conclusion**
The model achieved an accuracy of 76.93%. While it performs well in identifying non-toxic comments, it struggles with distinguishing less toxic and highly toxic comments. The results highlight the challenges of class imbalance and borderline classifications in toxicity detection. Further improvements can be achieved through threshold tuning, fine-tuning the model, and addressing class imbalance.
