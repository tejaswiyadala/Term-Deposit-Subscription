# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 11:08:16 2021

@author: kamar
"""

from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql.functions import avg, col, length
import matplotlib.pyplot as plt

conf = SparkConf().setAppName("Project App").setMaster("local")
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

from pyspark.sql import SQLContext

sql_context = SQLContext(sc)

PATH = r"D:\Kee cllg\spring 2021\CIS 5367 Machine ln\Project\Project submission"

bank_data=sql_context.read.load("%s/bank-full.csv" % PATH,    
                      format='com.databricks.spark.csv', 
                      header='true', 
                      inferSchema='true')

print(bank_data.head(5))


#Output Variable is y. Change column "y" to "Target"

import pyspark.sql.functions as F

bank_data = bank_data.select( '*', F.col('y').alias('Target') ).drop('y')


"""Summary statistics of the dataframe 
"""
bank_data.summary().show()
bank_data.summary().show(truncate=False)
bank_data.describe().show()
bank_data.show()

"""Print schema of the bank data"""

bank_data.printSchema() 

  
"""Check the value count for each unique value in the target column
"""

bank_data.select("Target").distinct().show()
deposit = bank_data.rdd.map(lambda x: x[16]).countByValue()


"""Summary statistics for numeric variables
"""
numerical_features=bank_data.select("age","balance","day","duration","campaign","pdays","previous")
numerical_features.describe().show()
print(numerical_features.show(5))

# Checking distribution of all numeric features in the datase

numeric_features = [t[0] for t in bank_data.dtypes if t[1] == 'int']
bank_data.select(numeric_features).describe().toPandas().transpose()



"""Summary statistics for Categorical variables
"""
Categorical_features=bank_data.select("job","marital","education","default","housing","loan","month","poutcome")
Categorical_features.describe().show()
print(Categorical_features.show(5))

"""Pair Plots
"""

import pandas as pd
from pandas.plotting._misc import scatter_matrix

numeric_data = bank_data.select("age","balance","day","duration","campaign","pdays","previous").toPandas()
axs = pd.plotting.scatter_matrix(numeric_data, figsize=(8, 8));
n = len(numeric_data.columns)
for i in range(n):
    v = axs[i, 0]
    v.yaxis.label.set_rotation(0)
    v.yaxis.label.set_ha('right')
    v.set_yticks(())
    h = axs[n-1, i]
    h.xaxis.label.set_rotation(90)
    h.set_xticks(())

    
"""
Correlations between independent variables
"""
from pyspark.sql.functions import when

#Replace yes with  1 and no with 0

bank_data= bank_data.withColumn('deposit', when(bank_data['target']=='yes',1).otherwise(0))
numeric_data = bank_data.select("age","balance","day","duration","campaign","pdays","previous","deposit").toPandas()

import seaborn as sns

import numpy as np
matrix = np.triu(numeric_data.corr())
sns.heatmap(numeric_data.corr(), annot=True, mask=matrix)


"""Show number of customers that have signed up term deposit vs those 
that did not  OR Number of respondents by the deposit subscription results
"""
numOfSignUps = bank_data.groupBy("deposit").count().show()


numOfSignUps = bank_data.groupBy("target").count().collect()

numOfSignUps_plot = bank_data.rdd.map(lambda x: x[16]).countByValue()
plt.bar(list(numOfSignUps_plot.keys()), numOfSignUps_plot.values(), color='g')


"""Percentage of category of deposit
"""

import pyspark.sql.functions as F

bank_data.groupby('target').agg(
    (F.count('target')).alias('count'),
    (F.count('target') / bank_data.count()).alias('percentage')
).show()



"""
#Number of respondants by Job Type
"""
                                                            
job_type = bank_data.groupBy("job").count().show()

plt.figure(figsize=(20,10))

job_type_plot = bank_data.rdd.map(lambda x: x[1]).countByValue()
plt.bar(list(job_type_plot.keys()), job_type_plot.values(), color='g')
plt.xticks(rotation=90)



"""
Number of respondents by marital status

"""
marital_status = bank_data.groupBy("marital").count().show()

plt.figure(figsize=(20,10))

marital_status_plot = bank_data.rdd.map(lambda x: x[2]).countByValue()
plt.bar(list(marital_status_plot.keys()), marital_status_plot.values(), color='green')


"""Number of respondents by level of education
"""

education = bank_data.groupBy("education").count().show()

plt.figure(figsize=(20,10))

education_plot = bank_data.rdd.map(lambda x: x[3]).countByValue()
plt.bar(list(education_plot.keys()), education_plot.values(), color='green')


"""
Number of respondents by the type of contact
"""
contact = bank_data.groupBy("contact").count().show()

plt.figure(figsize=(20,10))

contact_plot = bank_data.rdd.map(lambda x: x[8]).countByValue()
plt.bar(list(contact_plot .keys()), contact_plot .values(), color='green')


"""
Number of respondents by the outcome of the previous campaign

"""

poutcome = bank_data.groupBy("poutcome").count().show()

plt.figure(figsize=(20,10))

poutcome_plot = bank_data.rdd.map(lambda x: x[15]).countByValue()
plt.bar(list(poutcome_plot .keys()), poutcome_plot .values(), color='g')


"""Number of respondents by month

"""

month = bank_data.groupBy("month").count().show()

plt.figure(figsize=(20,10))

month_plot = bank_data.rdd.map(lambda x: x[10]).countByValue()
plt.bar(list(month_plot .keys()), month_plot .values(), color='green')



# Percentage by month

import pyspark.sql.functions as F

bank_data.groupby('month').agg(
    (F.count('target')).alias('count'),
    (F.count('target') / bank_data.count()).alias('percentage')
).show()


"""
Age Categories
 
"""
 
 
age = bank_data.groupBy("age").count().show()

plt.figure(figsize=(20,10))

age_plot = bank_data.rdd.map(lambda x: x[0]).countByValue()
plt.bar(list(age_plot .keys()), age_plot .values(), color='green')


""" Campaign """

campaign = bank_data.groupBy("campaign").count().show()

plt.figure(figsize=(20,10))

campaign_plot = bank_data.rdd.map(lambda x: x[12]).countByValue()
plt.bar(list(campaign_plot .keys()), campaign_plot .values(), color='green')

#Replace yes with  1 and no with 0
from pyspark.sql import functions as F
from pyspark.sql.functions import when

bank_data= bank_data.withColumn('deposit', when(bank_data['target']=='yes',1).otherwise(0))
bank_data = bank_data.drop('target')
bank_data.show()


"""Bivariate Analysis
"""
  
# Job versus deposit
bank_data.crosstab('job', 'deposit').show()

import pandas as pd
job = bank_data.crosstab('job', 'deposit').show()



# deposit versus marital status

marital_deposit=bank_data.crosstab('marital', 'deposit').show()

# Crosstabing categorical variables with the label
import pyspark.sql.functions as f

categorical_features = [t[0] for t in bank_data.dtypes if t[1] != 'int']

for f in categorical_features:
  bank_data.crosstab(f,'Deposit').show()
  
# count of marital status versus the deposit 
bank_data.groupBy("marital","deposit").count().show()

"""
Age Categories Versus deposit
"""
age = bank_data.groupBy("age","deposit").count().show()
plt.figure(figsize=(20,10))
age_plot = bank_data.rdd.map(lambda x: x[0]).countByValue()
plt.bar(list(age_plot .keys()), age_plot .values(), color='green')


""" Compaign Versus Target
"""

compaign = bank_data.groupBy("campaign","deposit").count().show()
plt.figure(figsize=(20,10))
compaign_plot = bank_data.rdd.map(lambda x: x[11]).countByValue()
plt.bar(list(compaign_plot .keys()), compaign_plot .values(), color='green')


#Down sampling
major_df = bank_data.filter(col("deposit") == 0)
minor_df = bank_data.filter(col("deposit") == 1)
ratio = int(major_df.count()/minor_df.count())
print("ratio: {}".format((ratio),2))

sampled_majority_df = major_df.sample(False, 1/ratio)
combined_df_2 = sampled_majority_df.unionAll(minor_df)
combined_df_2.show()

combined_df_2.count()
combined_df_2.groupby('deposit').count().show()
