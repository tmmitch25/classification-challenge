# Spam Detection Classification Challenge

## Background

In this project, I worked as a data scientist for an Internet Service Provider (ISP) to improve the email filtering system for its customers. The task was to develop a supervised machine learning (ML) model that could accurately classify emails as spam or not spam. The goal was to filter out spam emails from customer inboxes using two machine learning models: Logistic Regression and Random Forest. 

The provided dataset contained email information with two possible classifications: spam (1) and not spam (0). I was tasked with building and evaluating both models to determine which performed better at detecting spam emails.

## Steps Completed

### Step 1: Split the Data into Training and Testing Sets

python
import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data into a Pandas DataFrame
url = "https://static.bc-edx.com/ai/ail-v-1-0/m13/challenge/spam-data.csv"
data = pd.read_csv(url)

# Create the labels set (y) from the "spam" column
y = data['spam']

# Create the features DataFrame (X) from the remaining columns
X = data.drop(columns='spam')

# Check the balance of the labels variable
y.value_counts()

# Split the data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


### Step 2: Scale the Features

python
from sklearn.preprocessing import StandardScaler

# Create an instance of StandardScaler
scaler = StandardScaler()

# Fit the scaler with the training data and transform the training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform the testing data
X_test_scaled = scaler.transform(X_test)


### Step 3: Create a Logistic Regression Model

python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Create and fit a logistic regression model
logreg = LogisticRegression(random_state=1)
logreg.fit(X_train_scaled, y_train)

# Make predictions on the testing data
y_pred_logreg = logreg.predict(X_test_scaled)

# Evaluate the model's performance
logreg_accuracy = accuracy_score(y_test, y_pred_logreg)
print("Logistic Regression Accuracy:", logreg_accuracy)


### Step 4: Create a Random Forest Model

python
from sklearn.ensemble import RandomForestClassifier

# Create and fit a random forest model
rf = RandomForestClassifier(random_state=1)
rf.fit(X_train_scaled, y_train)

# Make predictions on the testing data
y_pred_rf = rf.predict(X_test_scaled)

# Evaluate the model's performance
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", rf_accuracy)


### Step 5: Evaluate the Models

- **Which model performed better?**  
  Based on the accuracy scores printed by the code above, you can compare the logistic regression and random forest models to determine which performed better.

- **How does that compare to your prediction?**  
  Reflect on your initial prediction regarding which model would perform better and compare it to the final results.

## Final Evaluation

- The model with the highest accuracy was selected as the better performing model.
- The results were compared to my initial prediction to assess how accurate my expectations were.

## Files

- spam_detector.ipynb: The Jupyter notebook containing all the code, model training, evaluation, and analysis.
- spam-data.csv: The dataset used for training and testing the models.

## Requirements Completed
1. **Data Splitting**: The dataset was correctly split into training and testing datasets using train_test_split().
2. **Feature Scaling**: StandardScaler was applied to normalize the features.
3. **Logistic Regression Model**: A logistic regression model was built, trained, and evaluated.
4. **Random Forest Model**: A random forest model was built, trained, and evaluated.
5. **Model Comparison**: The performance of both models was evaluated and compared based on accuracy.
