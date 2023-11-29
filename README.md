# Cricket-Data-AnalysisI developed a robust machine learning model to predict the margin of victory in IPL cricket matches using the Support Vector Regression (SVR) algorithm. Leveraging the pandas and scikit-learn libraries in Python, I began by loading the dataset from a CSV file and performed data cleaning by removing columns with a significant number of missing values.

For feature selection, I chose relevant attributes such as match details, team information, venue, toss details, and players involved. Utilizing LabelEncoder, I encoded categorical variables to facilitate the SVR model's training. To handle missing data, I employed SimpleImputer with a mean strategy, ensuring a comprehensive dataset for training.

Following preprocessing, I split the data into training and testing sets using the train_test_split function. To enhance model performance, I applied feature scaling using StandardScaler. The SVR model was instantiated and trained on the scaled training data, and predictions were made on the test set.

Model evaluation was conducted using key regression metrics: Mean Squared Error (MSE), R-squared (R²), Mean Absolute Error (MAE), and Mean Absolute Percentage Error (MAPE). These metrics provided insights into the model's accuracy and ability to generalize to unseen data.

Additionally, I visualized the model's predictions against actual values through a scatter plot using matplotlib. This graphical representation offered a clear overview of the model's predictive capabilities and showcased the relationship between predicted and actual outcomes.

In summary, this project not only demonstrated my proficiency in data preprocessing, feature engineering, and model training but also highlighted my ability to evaluate and communicate the model's performance effectively. The concise and insightful visualizations added a layer of interpretability to the results, showcasing my skills in both machine learning implementation and data presentation.
