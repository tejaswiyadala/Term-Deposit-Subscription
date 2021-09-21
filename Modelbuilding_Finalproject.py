# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 12:08:57 2021

@author: mumta
"""


# Importing required libraries and modules
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SQLContext
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, when
import pyspark.sql.functions as F
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import LogisticRegression
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from sklearn.metrics import roc_curve, auc
from pyspark.sql.functions import isnan, when, count, col



conf = SparkConf().setAppName(" Classifying Term Deposit Subscriptions for a bank Project").setMaster("local")
sc = SparkContext(conf=conf)
spark = SparkSession(sc)
sql_context = SQLContext(sc)

PATH = r"C:/Users\mumta/Documents/Mumtaz- College/Course/Spring 2021/CIS-5367 - MACHINE LEARNING/Project"

bank_data=sql_context.read.load("%s/bank-full.csv" % PATH,    
                      format='com.databricks.spark.csv', 
                      header='true', 
                      inferSchema='true')

print(bank_data.head(5))
bank_data.count()


#Output Variable is y. Changing column "y" to "Target" for further processing

bank_data=bank_data.withColumnRenamed("y","target")
bank_data.show()


### Get count of nan or missing values in dataset

bank_data.select([count(when(isnan(c), c)).alias(c) for c in bank_data.columns]).show()

# Percentage of each category of target variable

bank_data.groupby('target').agg(
    (F.count('target')).alias('count'),
    (F.count('target') / bank_data.count()).alias('percentage')
).show()


# UpSampling to handle inbalanced dataset

major_df = bank_data.filter(col("target") == "no")
minor_df = bank_data.filter(col("target") == "yes")
ratio = int(major_df.count()/minor_df.count())
print(ratio)

df_b_oversampled = minor_df.sample(True,fraction=float(ratio), seed=1)
combined_df = major_df.unionAll(df_b_oversampled)

combined_df.count()

print (" Total count of dataset after UpSampling", combined_df.count())

# Count of each ctegory in target class 
combined_df.groupby('target').count().show()


# Feature Selection for the model

# We kept all of the variables in our model. Except variables day and month which are not really useful, we will remove these two columns
# Selecting variables or features to be used for further processing

combined_df=combined_df.select('age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'campaign', 'pdays', 'previous', 'poutcome', 'target')
combined_df.show(5)
combined_df.count()
cols=combined_df.columns


# Handling categorical columns for our model
# String Indexing, One Hot Encoding and Vector Assembling:
# Convert categorical features to numeric features using One hot encoding
#SELECTING CATEGORICAL COLUMNS ONLY

categoricalColumns = ['job','marital','education','default','housing','loan','poutcome']

#CREATING AN EMPTY LIST FOR PIPELINE AND ASSEMBLER

stages = []

#APPLYING FOR LOOP TO INDEX AND ENCODE ALL THE SELECTED COLUMNS
#APPLYING STRING INDEXER TO ALL THE CATEGORICAL COLUMNS AND STORING IT IN A NEW COLUMN WITH +INDEXED
#APPLYING ONE HOT ENCODER TO ALL THE INDEXED COLUMNS AND STORING IT IN A NEW COLUMN WITH +ENCODED


for categoricalCol in categoricalColumns:
    
    stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
    encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    stages += [stringIndexer, encoder]
    
    
#INDEXING PREDICTOR COLUMN 'DEPOSIT' AS LABEL AND FEATURES    

target_stringIndex = StringIndexer(inputCol = 'target', outputCol = 'label')

#CREATING STAGES FOR BOTH NUMERICAL AND CATEGORICAL COLUMNS

stages += [target_stringIndex]


# Transform all features into a vector using VectorAssembler

# ADDING BOTH To ASSEMBLER


numericalColumns = ['age', 'balance', 'campaign', 'pdays', 'previous']
assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericalColumns


# VECTORIZING TO CREATE A NEW FEATURES COLUMN WITH INDEXED AND ENCODED VALUES

assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]


# Run the stages as a Pipeline. This puts the data through all of the feature transformations in a single call.
# COMBINING ALL THE STAGES INTO ONE, FITTING combined_df AND TRANSFORMING IT


pipeline = Pipeline(stages = stages)
pipelineModel = pipeline.fit(combined_df)
df = pipelineModel.transform(combined_df)
df.show()
df.count()
df.select("features").show(truncate=0)


# Normalization using min max scaler:

# The feature vector has been finally normalized using the min-max scaler in pyspark and transformed as below:
# Apply Min-Max normalisation on each attribute using MinMaxScaler  

scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")
scalerModel = scaler.fit(df)
scaledBankData = scalerModel.transform(df)
scaledBankData.select("features", "scaledFeatures").show(truncate=0)

#To check values convert to PANDAS dataframe

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)
scaledBankData.select('scaledFeatures').toPandas().head(1)



# Model Building 

#The dataset has been split at 70:30. 70% of the dataset has been kept for training the supervised learning models and 30% of the dataset has been kept for testing the dataset.

train, test = scaledBankData.randomSplit([0.7, 0.3], seed = 742)
print("Training Dataset Count: " + str(train.count()))
print("Test Dataset Count: " + str(test.count()))


# Logistic Regression modelling


lr = LogisticRegression(featuresCol = 'scaledFeatures', labelCol = 'label', maxIter=10)
LRModel = lr.fit(train)


# predicting on testing set

LRpredictions = LRModel.transform(test)
LRpredictions.show(truncate=False)


# Evaluation Metrics for Testing Set

# USING BINARY CLASS EVALUATOR FOR TEST AREA UNDER ROC CALCULATION

evaluator = BinaryClassificationEvaluator()
print('Test Area Under ROC', evaluator.evaluate(LRpredictions))


# ROC curve

PredAndLabels_lr       = LRpredictions.select("probability", "label").collect()
PredAndLabels_list_lr  = [(float(i[0][0]), 1.0-float(i[1])) for i in PredAndLabels_lr]

 
y_test_lr = [i[1] for i in PredAndLabels_list_lr]
y_score_lr = [i[0] for i in PredAndLabels_list_lr]
 
fpr1, tpr1, _ = roc_curve(y_test_lr, y_score_lr)
roc_auc_lr = auc(fpr1, tpr1)
 
plt.figure(figsize=(8,8))
plt.plot(fpr1, tpr1, label='ROC curve (area = %0.2f)' % roc_auc_lr)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontsize=20)
plt.ylabel('True Positive Rate',fontsize=20)
plt.title('ROC Curve - Logistic Regression',fontsize=20)
plt.legend(loc="lower right")
plt.show()


# PRINTING ONLY LABEL AND PREDICTION FOR ACCURACY CALCULATION

accdf=LRpredictions.select("label","prediction").show(5)


# Metric Evaluators for the model 
# Calculating evaluation metrics of the Logistic regression model


# Calculating the True Positive, True Negative, False Positive and False Negative categories
TrueNegative = LRpredictions.filter('prediction = 0 AND label = prediction').count()
TruePositive = LRpredictions.filter('prediction = 1 AND label = prediction').count()
FalseNegative = LRpredictions.filter('prediction = 0 AND label <> prediction').count()
FalsePositive = LRpredictions.filter('prediction = 1 AND label <> prediction').count()
print("TN,FP,FN,TP",TrueNegative,FalsePositive,FalseNegative,TruePositive)

# Accuracy, precision, recall, f1-score

accuracy = (TrueNegative + TruePositive) / (TrueNegative + TruePositive + FalseNegative + FalsePositive)
precision = TruePositive / (TruePositive + FalsePositive)
recall = TruePositive / (TruePositive + FalseNegative)
F =  2 * (precision*recall) / (precision + recall)
print('\n Precision: %0.3f' % precision)
print('\n Recall: %0.3f' % recall)
print('\n Accuracy: %0.3f' % accuracy)
print('\n F1 score: %0.3f' % F)


# Area under ROC curve
evaluator = BinaryClassificationEvaluator()
print('Test Area Under ROC', evaluator.evaluate(LRpredictions))

# CONFUSION MATRIX

# check labels for target class
 
class_temp = LRpredictions.select("label").groupBy("label")\
                        .count().sort('count', ascending=False).toPandas()
class_temp = class_temp["label"].values.tolist()

predandlabel=LRpredictions.select( 'label', 'prediction').rdd
metrics = MulticlassMetrics(predandlabel)
print(metrics.confusionMatrix())

cm=metrics.confusionMatrix().toArray()
print(cm)

# Forming the confusion matrix plot

f, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt=".1f", linewidths=.5, ax=ax,cmap=plt.cm.Blues)
sns.set(font_scale=2.5)
plt.title("Confusion Matrix", fontsize=20)
plt.subplots_adjust(left=0.15, right=0.99, bottom=0.15, top=0.99)
ax.set_yticks(np.arange(cm.shape[0]) + 0.5, minor=False)
ax.set_xticks(np.arange(cm.shape[0]) + 0.5, minor=False)

plt.ylabel('Predicted label',size = 30)
plt.xlabel('Actual label',size = 30)
plt.show()


###############################################

# Random Forest modelling

rf = RandomForestClassifier(featuresCol = 'scaledFeatures', labelCol = 'label', maxDepth=10)
RFModel = rf.fit(train)


# predicting on testing set

rfpredictions = RFModel.transform(test)
rfpredictions.show(truncate=False)


# Evaluation Metrics for Testing Set

# USING BINARY CLASS EVALUATOR FOR TEST AREA UNDER ROC CALCULATION

evaluator = BinaryClassificationEvaluator()
print('Test Area Under ROC', evaluator.evaluate(rfpredictions))


# Plot the ROC curve 


PredAndLabels_rf          = rfpredictions.select("probability", "label")
PredAndLabels_collect_rf   = PredAndLabels_rf.collect()
PredAndLabels_list_rf     = [(float(i[0][0]), 1.0-float(i[1])) for i in PredAndLabels_collect_rf]
 
y_test_rf = [i[1] for i in PredAndLabels_list_rf]
y_score_rf = [i[0] for i in PredAndLabels_list_rf]
 
fpr2, tpr2, _ = roc_curve(y_test_rf, y_score_rf)
roc_auc_rf = auc(fpr2, tpr2)
 
plt.figure(figsize=(8,8))
plt.plot(fpr1, tpr1, label='ROC curve (area = %0.2f)' % roc_auc_rf)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontsize=20)
plt.ylabel('True Positive Rate',fontsize=20)
plt.title('ROC Curve - Random Forest',fontsize=20)
plt.legend(loc="lower right")
plt.show()



#PRINTING ONLY LABEL AND PREDICTION FOR ACCURACY CALCULATION

accdf=rfpredictions.select("label","prediction").show(5)


# Metric Evaluators

#Calculating metrics of the Random Forest model


TrueNegative_rf = rfpredictions.filter('prediction = 0 AND label = prediction').count()
TruePositive_rf = rfpredictions.filter('prediction = 1 AND label = prediction').count()
FalseNegative_rf = rfpredictions.filter('prediction = 0 AND label <> prediction').count()
FalsePositive_rf= rfpredictions.filter('prediction = 1 AND label <> prediction').count()
print("TN,FP,FN,TP",TrueNegative_rf,FalsePositive_rf,FalseNegative_rf,TruePositive_rf)



accuracy = (TrueNegative_rf + TruePositive_rf) / (TrueNegative_rf + TruePositive_rf + FalseNegative_rf + FalsePositive_rf)
precision = TruePositive_rf / (TruePositive_rf + FalsePositive_rf)
recall = TruePositive_rf / (TruePositive_rf + FalseNegative_rf)
F =  2 * (precision*recall) / (precision + recall)
print('\n Precision: %0.3f' % precision)
print('\n Recall: %0.3f' % recall)
print('\n Accuracy: %0.3f' % accuracy)
print('\n F1 score: %0.3f' % F)
evaluator = BinaryClassificationEvaluator()
print('Test Area Under ROC', evaluator.evaluate(rfpredictions))

#CONFUSION MATRIX

predandlabel=rfpredictions.select( 'label', 'prediction').rdd
metrics = MulticlassMetrics(predandlabel)
print(metrics.confusionMatrix())


# PLOTTING HEATMAP OF ALL THE METRICS PARAMETERS

cm1=metrics.confusionMatrix().toArray()
f, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(cm1, annot=True, fmt=".1f", linewidths=.5, ax=ax,cmap=plt.cm.Blues)
sns.set(font_scale=2.5)
plt.title("Confusion Matrix", fontsize=20)
plt.subplots_adjust(left=0.15, right=0.99, bottom=0.15, top=0.99)
ax.set_yticks(np.arange(cm1.shape[0]) + 0.5, minor=False)
ax.set_xticks(np.arange(cm1.shape[0]) + 0.5, minor=False)

plt.ylabel('Predicted label',size = 30)
plt.xlabel('Actual label',size = 30)
plt.show()



# Creating a table for the evalution metrics for both logistic regression and random forest

binary_eval = BinaryClassificationEvaluator(labelCol = "label")
multi_eval_acc = MulticlassClassificationEvaluator(labelCol = "label", metricName = "accuracy")

# Create pandas data frame to store the result of our model metrics

model_metrics = pd.DataFrame(columns = ["Dataset", "Accuracy", "ROC"])

model_metrics = model_metrics.append(
    {
        "Dataset" : "Testing Set",
        "Code" : "Random Forest",
        "Accuracy" : multi_eval_acc.evaluate(rfpredictions),
        "ROC" : binary_eval.evaluate(rfpredictions)
    },
    ignore_index = True 
    
)

model_metrics = model_metrics.append(
    {
        "Dataset" : "Testing Set",
        "Code" : "Logistic Regression",
        "Accuracy" : multi_eval_acc.evaluate(LRpredictions),
        "ROC" : binary_eval.evaluate(LRpredictions)
    },
    ignore_index = True
)


model_metrics


# Random Forest gives better accuracy so we proceed with this model
# Exporting predictions to csv files for both the models

#Converting predicted values for target back to orginal yes or no


rfpredictions= rfpredictions.withColumn('prediction', when(rfpredictions['prediction']==1.0,"yes").otherwise("no"))
datafame_to_export=rfpredictions.select('age', 'job','marital','education','default','balance',
 'housing','loan','campaign','pdays','previous','poutcome','prediction')
datafame_to_export.toPandas().to_csv('CustomerDetails_RandonForestOutput.csv',index=False)



