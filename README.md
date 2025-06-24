### Project Title

**Author**
Lori Smith

#### Executive summary


This project is based on a web based e-commerce site that supports small business.  

#### Rationale
Why should anyone care about this question?

Small business needs a way to compete with large business.  Using some of the AI image tools helps them to have similar tools that big tech has.

This project matters because it empowers small, local businesses to compete with large e-commerce platforms like Amazon. By enabling customers to visually search for products available nearby, we help drive foot traffic and online orders to independent shops that often struggle with visibility online. For consumers, it provides a fast, intuitive way to find what they need locally—potentially reducing wait times and environmental impact from shipping.

From a broader social perspective, this type of platform strengthens local economies and fosters community connections. On a personal level, this project also enables me to gain experience with image-based machine learning, which is a step toward solving more complex problems involving 3D graphics and AI—my background and future goal.

#### Research Question
What are you trying to answer?

How can image classification be used to support visual product search in a local shopping platform like BuyIt.store, which connects small businesses with nearby customers?

#### Data Sources
What data will you use to answer you question?

I am using the Fashion Product Images Dataset available on Kaggle:

Dataset: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-datasetLinks to an external site.
Sample Classifier: https://www.kaggle.com/code/paramaggarwal/fashion-product-images-classifierLinks to an external site.

#### Methodology
What methods are you using to answer the question?  

I plan to implement and compare the following classification algorithms to determine which best classifies product images:

Decision Trees
Logistic Regression
K-Nearest Neighbors (KNN)
Support Vector Classifier (SVC)
Neural Networks

#### Results
What did your research find?

The initial model searches had mixed results.  I ran KNN, DecisionTree and Logistic.

![Description](images/incorrect_predictions.png)
Here were some of the results based on my initial tests.

I found that SVN takes too long to process after a half hour of waiting for a result. The most successful model in my initial tests was the KNN Classifier with a 72.5% accuracy.

![Description](images/initial_model_accuracies.png)

#### Next steps
What suggestions do you have for next steps?

A better dataset with more accurate labeling would help.  Also some of the data is shown worn on a user making it harder to categorize.  For example a man is wearing a jacket and a scarf and the image is classified as jacket when it should be classified as scarf.  Some of the mistakes that are being made are because of poor labeling. 

#### Outline of project

- [Link to notebook 1]()
- [Link to notebook 2]()
- [Link to notebook 3]()


##### Contact and Further Information