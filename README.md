# Understanding_and_Implementing_simple_and_multiple_linear_regression
Built a linear regression model to predict house prices from multiple features. Preprocessed categorical data, split into train/test sets, trained using sklearn, evaluated performance, visualized results, and interpreted feature impact for better real estate analysis.
Linear Regression for Housing Prices
Overview
This project demonstrates a complete workflow for predicting house prices using linear regression. Leveraging the Housing dataset, the key objective was to accurately estimate residential property value based on features such as area, bedrooms, bathrooms, stories, parking, amenities (main road access, guestroom, basement, hot water heating, air conditioning), preferred area, and furnishing status.​

Steps Performed
Data Preprocessing: Transformed categorical features into numerical form. Binary columns (yes/no) were encoded, and multi-class features like furnishing status were handled via one-hot encoding for better model interpretation.​

Train-Test Split: Data was split into 80% training and 20% testing to assess model performance on unseen cases.​

Model Building: Used scikit-learn’s LinearRegression to train a multiple linear regression model using all available predictors.​

Evaluation: The model was evaluated using MAE, MSE, and R², quantifying prediction accuracy and ability to explain variance in house prices.​

Visualization: Plotted predicted vs. actual prices for the area feature to visually assess alignment and model accuracy.​

Interpretation: Analyzed regression coefficients to understand the impact of each feature on price, providing actionable insights for real estate valuation.​

Results
The model was able to capture the relationships between features and house prices, offering good generalization performance. Evaluation metrics supported the effectiveness of preprocessing and modeling steps, and visualizations demonstrated close alignment between predicted and actual values. Feature coefficient analysis helped identify which attributes most strongly influenced pricing.

How to Run
Install required libraries: pandas, scikit-learn, matplotlib.

Place Housing.csv in your working directory.

Run the provided Python script.

License:
This project is for educational purposes. Data source: Housing database
