# Lab deliverable 2 : NERC task
_Mining Unstructured Data, Master in Data Science, Barcelona School of Informatics_

Javier Herrer Torres (javier.herrer.torres@estudiantat.upc.edu)
Ignacio Lloret Lorente (ignacio.lloret@estudiantat.upc.edu)

---

[GitHub repository](https://github.com/IgnacioLL/NERC-classification)

## Introduction

This report presents an exploration into Named Entity Recognition and Classification (NERC) tasks using machine learning techniques. The primary goal of this work is to develop a system capable of identifying and categorizing drug names within unstructured text data. 

The project focuses on implementing a Sequence Tagging approach using the B-I-O schema.

To accomplish the NERC task, the project adopts a structured approach consisting of feature extraction, model training, and classification. Feature extraction involves capturing relevant information about tokens and their contexts to inform the learning process. Various machine learning algorithms are explored for model training, including Conditional Random Fields (CRF) and Naive Bayes, with an emphasis on experimenting with different parameterizations and feature combinations to optimize performance.

For this we will play with different features extracted from the tokens in order to accomplish this task. We will also try different algorithms to try to improve the CRF model. We will mesure the improvements made by the feature extractor by predicting a test dataset containing entities that the model must predict its category. As the main metric we will use is the recall and F1 in order to measure how the model performs in an unbalanced target challenge. 

The deliverables of this project include a comprehensive report detailing the methodologies employed and insights gained from the experimentation process. Additionally, the report includes the extract features function, which forms a critical component of the feature extraction process, along with any relevant code snippets.

Overall, the objective of this project is to develop a proficient NERC system for drug name identification and classification, leveraging machine learning techniques to achieve accurate and efficient entity recognition within unstructured text data and test the importance of feature extraction in the context of NERC. 