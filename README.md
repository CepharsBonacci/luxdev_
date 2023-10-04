# luxdev_
Churnpredictionmodel

Churnpredictionmodel

1.# DATA COLLECTION I used mockaroo.com to generate my dataset for the churn prediction mockaroo does not require any programming skills for generating data,you just specify your features and the number of fields you want and it will automatically generate the data

2.# IMPORTATION OF LIBRARIES .I imported pyforest libraries i.e,matplotlib,seaborn,numpy and pandas

3.# EXPLORATORY DATA ANALYSIS I inspected df.isnull() to see the missing values and i found out that the data were missing NOT completely at Random (NMAR) since there were relationship between one feature and other features.Hence i could not impute using simple Imputers like mean,Median.Again.Hence the only option Was to drop the missing Values

4.# FEATURE ENGINEERING There were columns which were categorical hence the need to convert them to numerical.Therefore i did this

Encode categorical columns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

we will use OneHot To Encode Gender column and LabelEncoder to Encode customer_segment then convert last_purchase_date into datetime and get valuable values like year,month,day
convert churned column(target column) to numerical, False == 0, True == 1
dev['churned']=dev['churned'].replace({False: 0, True: 1})

Lets use LabelEncoder for 'customer_segment'
label_encoder = LabelEncoder() dev['customer_segment_encoded'] = label_encoder.fit_transform(dev['customer_segment'])

lets convert date to datetime into year, month, day
dev['last_purchase_date'] = pd.to_datetime(dev['last_purchase_date']) dev['year'] = pd.to_datetime(dev['last_purchase_date']).dt.year dev['month'] = pd.to_datetime(dev['last_purchase_date']).dt.month dev['day'] = pd.to_datetime(dev['last_purchase_date']).dt.day

Lets use OneHotEncoder for 'gender'
onehot_encoder = OneHotEncoder(sparse=False) gender_encoded = onehot_encoder.fit_transform(dev[['gender']]) gender_encoded_df = pd.DataFrame(gender_encoded, columns=['gender_' + str(int(i)) for i in range(gender_encoded.shape[1])])

Concatenate the encoded columns back to the original DataFrame
dev_encoded = pd.concat([dev, gender_encoded_df], axis=1)

Lets drop the original gender and customer_segment
dev_encoded = dev_encoded.drop(['gender', 'customer_segment', 'last_purchase_date'], axis=1)

Again,i used MinMaxScaler to scale our values to one scale (0-1)

5.# MODEL SELECTION I experimented RandomForestClassifier and XGBoost and XGBoost seems to be performing better model = XGBClassifier() model.fit(X_train, y_train)

6.EVALUATION METRIC I used accuracy_score since it was a classification problem
