#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pyforest
import warnings
warnings.filterwarnings("ignore")


# In[3]:


train = pd.read_csv('train.csv')


# In[4]:


train.head()


# In[5]:


train.shape


# In[6]:


train.info()


# In[7]:


train.duplicated().sum()


# In[8]:


# Set plot style
sns.set(style="whitegrid")

#  Histograms for Each Feature
train.hist(bins=20, figsize=(15, 12), color='skyblue')
plt.suptitle('Distribution of Each Feature', fontsize=16)
plt.tight_layout()
plt.show()


# In[9]:


# Correlation Heatmap
plt.figure(figsize=(12, 8))
corr_matrix = train.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap', fontsize=16)
plt.show() 


# In[10]:


# Pairplot (Selected Columns)
selected_columns = ['credit_score', 'income', 'loan_amount', 'interest_rate', 'default_risk_score']
sns.pairplot(train[selected_columns], diag_kind='hist')
plt.suptitle('Pairplot of Selected Features', y=1.02, fontsize=16)
plt.show()


# In[11]:


X = train.drop(columns=['default_risk_score'])


# In[12]:


X.head()


# In[13]:


Xcorr_matrix = X.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(Xcorr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap', fontsize=16)
plt.show()


# In[14]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
# Adding a constant for VIF calculation
X_vif = sm.add_constant(X)

# Calculating VIF for each feature
vif_data = pd.DataFrame()
vif_data['Feature'] = X_vif.columns
vif_data['VIF'] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]

#Display VIF values
vif_data.sort_values(by='VIF', ascending=False)


# In[15]:


# Plotting the boxplot for all numerical features
plt.figure(figsize=(15, 8))
sns.boxplot(data=train, orient='h')
plt.title('Boxplot of All Features')
plt.xlabel('Value')
plt.show()


# In[16]:


#checking if log transformation is needed
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(train['default_risk_score'], bins=30, kde=True)
plt.title("Distribution of Default Risk Score (Before Log Transform)")
plt.show()


# In[17]:


#apply log transformation to reduce skewness and handle outliers
train['default_risk_score'] = np.log(train['default_risk_score'] + 1)


# In[18]:


#check if the log transformation has been applied
sns.histplot(np.log(train['default_risk_score'] + 1), bins=30, kde=True)
plt.title("Distribution of Default Risk Score (After Log Transform)")
plt.show()


# In[19]:


train.head(10)


# In[20]:


y = train['default_risk_score']


# In[21]:


# 1. Debt-Related Features
train['Loan_to_Income_Ratio'] = train['loan_amount'] / train['income']
train['Savings_to_Income_Ratio'] = train['savings_balance'] / train['income']
train['Debt_Service_Coverage_Ratio'] = train['income'] / (train['loan_amount'] * train['interest_rate'] / 100)
train['Debt_Age_Ratio'] = train['debt_to_income_ratio'] / train['age']

# 2. Credit & Loan Features
train['Credit_Age'] = train['age'] - train['employment_years']
train['Loan_to_Savings_Ratio'] = train['loan_amount'] / train['savings_balance']
train['Effective_Interest_Rate'] = train['interest_rate'] / train['loan_term']

# 3. Risk-Related Features
train['Income_Risk_Score'] = train['income'] / (train['loan_amount'] * train['loan_term'])
train['Risk_Per_Year'] = train['default_risk_score'] / train['loan_term']

# 4. Interaction Features
train['Credit_Income_Interaction'] = train['credit_score'] * train['income']
train['Loan_Term_Employment_Interaction'] = train['loan_term'] * train['employment_years']

# 5. Categorical Binning
train['Age_Group'] = pd.cut(train['age'], bins=[18, 30, 50, 70], labels=['Young', 'Middle-aged', 'Senior'])
train['Income_Bracket'] = pd.qcut(train['income'], q=3, labels=['Low', 'Medium', 'High'])
train['Loan_Amount_Bracket'] = pd.qcut(train['loan_amount'], q=3, labels=['Small', 'Medium', 'Large'])

# 6. Aggregated Features
train['Total_Financial_Health'] = train['income'] + train['savings_balance'] - (train['loan_amount'] * train['debt_to_income_ratio'])
train['Net_Income_After_Loan'] = train['income'] - (train['loan_amount'] * train['interest_rate'] / 100)

# 7. Polynomial Features (Example: 2nd-degree interaction)
train['Credit_Age_Squared'] = train['Credit_Age'] ** 2
train['Income_Loan_Interaction'] = train['income'] * train['loan_amount']

# 8. Log-Transformations (For Highly Skewed Data)
train['Log_Income'] = np.log1p(train['income'])
train['Log_Savings_Balance'] = np.log1p(train['savings_balance'])
train['Log_Loan_Amount'] = np.log1p(train['loan_amount'])


# In[26]:


# Dimensionality Reduction with PCA (to preserve 95% of variance)
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
print("Number of PCA components selected:", pca.n_components_)


# In[ ]:


# Explained variance ratio of each principal component
explained_variance = pca.explained_variance_ratio_

# Cumulative explained variance
cumulative_variance = explained_variance.cumsum()

# Plot the variance explained by each component
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker="o", linestyle="--")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Explained Variance vs. Number of Components")
plt.grid(True)
plt.show()


# In[ ]:


# Model Selection and Hyperparameter Tuning Using Ridge Regression
# Ridge (L2 regularization) can help control overfitting.
ridge = Ridge()
param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(ridge, param_grid, cv=3, scoring='r2')  # Using 3-fold CV due to small data size
grid_search.fit(X_train_pca, y_train)
best_alpha = grid_search.best_params_['alpha']
print("Best alpha from grid search:", best_alpha)
# Train the final Ridge model with the best hyperparameter
best_ridge = Ridge(alpha=best_alpha)
best_ridge.fit(X_train_pca, y_train)


# In[23]:


# Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# In[24]:


#✅ Benchmark Performance: If Ridge Regression improves the R² or reduces RMSE significantly, you have evidence that regularization is needed.
#✅ Overfitting Detection: If Linear Regression performs well but Ridge does worse, your model might be over-regularized.
#✅ Feature Engineering Justification: If the baseline does poorly, it validates the need for feature engineering or dimensionality reduction.
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Initialize and train a simple Linear Regression model
baseline_model = LinearRegression()
baseline_model.fit(X_train, y_train)

# Predictions
y_train_pred = baseline_model.predict(X_train)
y_test_pred = baseline_model.predict(X_test)

# Evaluation
train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Baseline Model Performance:")
print(f"Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
print(f"Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")


# In[27]:


# Standardize the Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) #calculates the mean and standard deviation from x_train
X_test_scaled = scaler.transform(X_test)


# In[29]:


# Train Ridge Regression with a default alpha
ridge_model = Ridge(alpha=1.0)  
ridge_model.fit(X_train_scaled, y_train)

# Predictions
y_train_ridge_pred = ridge_model.predict(X_train_scaled)
y_test_ridge_pred = ridge_model.predict(X_test_scaled)

# Performance Metrics
train_rmse_ridge = mean_squared_error(y_train, y_train_ridge_pred, squared=False)
test_rmse_ridge = mean_squared_error(y_test, y_test_ridge_pred, squared=False)
train_r2_ridge = r2_score(y_train, y_train_ridge_pred)
test_r2_ridge = r2_score(y_test, y_test_ridge_pred)


# In[30]:


print("🔹 Baseline Model (Linear Regression)")
print(f"Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
print(f"Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")

print("\n🔹 Ridge Regression (Standardized Data)")
print(f"Train RMSE: {train_rmse_ridge:.4f}, Test RMSE: {test_rmse_ridge:.4f}")
print(f"Train R²: {train_r2_ridge:.4f}, Test R²: {test_r2_ridge:.4f}")


# In[28]:


X_const = sm.add_constant(X) #ensures the model has an intercept
ols_model = sm.OLS(y, X_const).fit()
print(ols_model.summary())


# In[ ]:


train.describe().transpose()


# In[ ]:


# Extract R² scores for each alpha value
r2_scores = grid_search.cv_results_['mean_test_score']
alphas = param_grid['alpha']

# Plot
plt.figure(figsize=(8, 5))
plt.plot(alphas, r2_scores, marker='o', linestyle='--', color='b')
plt.xscale('log')  # Log scale for better visualization
plt.xlabel("Alpha (log scale)")
plt.ylabel("Mean R² Score")
plt.title("R² Score vs. Ridge Alpha")
plt.grid(True)
plt.show()


# In[ ]:


# Model Evaluation and Test Predictions

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

y_pred = best_ridge.predict(X_test_pca)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)  
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R-squared Score:", r2)
print("Root Mean Squared Error (RMSE):", rmse)


# In[ ]:


# Visualizing Predictions
import matplotlib.pyplot as plt

# Scatter plot of Actual vs. Predicted values
plt.figure(figsize=(7,5))
plt.scatter(y_test, y_pred, color="blue", alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r', lw=2)  # Perfect fit line
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted Values (Ridge Regression)")
plt.grid()
plt.show()


# #### Running Actual Test File

# In[31]:


# Load the test data
test_df = pd.read_csv("test.csv")


# In[37]:


test_df.head()


# In[ ]:


# Applying the same feature engineering
test_df['Loan_to_Income_Ratio'] = test_df['loan_amount'] / test_df['income']
test_df['Savings_to_Income_Ratio'] = test_df['savings_balance'] / test_df['income']
test_df['Debt_Service_Coverage_Ratio'] = test_df['income'] / (test_df['loan_amount'] * test_df['interest_rate'] / 100)
test_df['Credit_Age'] = test_df['age'] - test_df['employment_years']

# Applying log transformation
import numpy as np
test_df['Log_Income'] = np.log1p(test_df['income'])
test_df['Log_Savings_Balance'] = np.log1p(test_df['savings_balance'])
test_df['Log_Loan_Amount'] = np.log1p(test_df['loan_amount'])

# Drop unnecessay columns
feature_columns = X_train.columns  # Ensure test data has the same features as training data
test_df = test_df[feature_columns]


# In[ ]:


# Load the saved scaler 
scaler = StandardScaler()
scaler.fit(X_train)  # Fit on training data only!

# Transform the test data
test_df_scaled = scaler.transform(test_df)


# In[ ]:


# Step 6: Making Predictions Using the Trained Model
test_predictions = ridge_model.predict(test_df_scaled)

# Step 7: Converting Predictions to a DataFrame for Display
test_results = pd.DataFrame({'Predicted_Default_Risk_Score': test_predictions})

# Step 8: Print First Few Predictions
print("🔹 First 10 Predictions:")
print(test_results.head(10))


# In[39]:


rmse = mean_squared_error(y_test, test_predictions, squared=False)
r2 = r2_score(y_test, test_predictions)

print(f"Test RMSE: {rmse:.4f}")
print(f"Test R² Score: {r2:.4f}")


# In[43]:


test_results = pd.DataFrame(test_predictions, columns=["Predicted_Default_Risk_Score"])
test_results.to_csv("ridge_predictions.csv", index=False)

