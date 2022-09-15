# ML_Inference
Machine learning inférence unsing pytorch and deployed on a local server using flask

## Models

### Image classification

 - Vision Transformer from google, state of the art model for image recognition
 - densenet model, deep learning aproach for the classification model. Ligther than the ViT model
 
### Text classification
I choose to use two robberta models

 - robberta base twitter sentiment, trained on annotated tweets
 - setimennt robberta large
 
I wanted to see the difference on inference on two diffrent scale of machine learning models, as a result, for simple sentences the larger model is unessesary beacause it has the same performances as the twitter trained one.

## How to run
Download the requirered package with the *requirement.txt* file

Run the *app.py* on your terminal and paste the local adress to the internet browser
