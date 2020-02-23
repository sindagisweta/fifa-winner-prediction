# initialize sparkSession
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[*]").appName("FIFA World Cup Predictor").getOrCreate()


# Library imports
import matplotlib.pyplot as plt

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import StringIndexer

from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorIndexer

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

#Read dataset from HDFS file location
dataset_old = spark.read.csv('hdfs://data/Result.csv',inferSchema=True, header =True)
dataset = spark.read.csv('hdfs://data/WorldCupMatches.csv',inferSchema=True, header =True)

#Print schema variables
dataset.printSchema()

# Data cleaning 
# Lable encoding - Converting string to numerical by assinging dummy values

indexer = StringIndexer(inputCol="Stadium", outputCol="Stadium_code")
indexed = indexer.fit(dataset).transform(dataset)

indexer = StringIndexer(inputCol="Stage", outputCol="Stage_code")
indexed = indexer.fit(indexed).transform(indexed)

indexer = StringIndexer(inputCol="City", outputCol="City_code")
indexed = indexer.fit(indexed).transform(indexed)

indexer = StringIndexer(inputCol="Home Team Name", outputCol="Home_Team_code")
indexed = indexer.fit(indexed).transform(indexed)

indexer = StringIndexer(inputCol="Away Team Name", outputCol="Away_Team_code")
indexed = indexer.fit(indexed).transform(indexed)

indexer = StringIndexer(inputCol="Referee", outputCol="Referee_code")
indexed = indexer.fit(indexed).transform(indexed)

indexer = StringIndexer(inputCol="Assistant 1", outputCol="Assistant_1_code")
indexed = indexer.fit(indexed).transform(indexed)

indexer = StringIndexer(inputCol="Assistant 2", outputCol="Assistant_2_code")
indexed = indexer.fit(indexed).transform(indexed)

indexed.show()


# Generate Dependent variable by coparing the score of Home & Away Team
# If Home Team score is greater than Away Team score variable "winner" will be sent to 1 
# Else will be set to 0

from pyspark.sql import functions as f
indexed = indexed.withColumn('winner', f.when(f.col('Home Team Goals') > f.col("Away Team Goals"), 1).otherwise(0))
indexed.show()

# Conver the Spark dataframe to Pandas dataframe for more analysis
pandas_df = indexed.toPandas()

# Data summary - to find some insights like Count, Mean, High, low, Avg of each columns
summary = pandas_df.describe(include='all')
summary

# Print correlation matrix to find relationship between feature and dependent variable
# Give the score value how the change in variable will affect the outcome
# Score ranges between 0 to 1. The higher the score the more the correlation(dependent)
pandas_df.corr()

# Generate winningTeam column by comparing the score of Home and Away team
pandas_df['winningTeam'] =  np.where(pandas_df['Home Team Goals'] > pandas_df['Away Team Goals'] , pandas_df['Home Team Name'], pandas_df['Away Team Name'])

# Generate winningTeam column by comparing the score of Home and Away team
winners_avg = pandas_df['winningTeam'].value_counts().to_frame()
winners_avg.columns

#Bar graph for Exploratory analysis
plt.bar(winners_avg.index, winners_avg['winningTeam'] ,align='center')

#Create new Dataframe to hold only dependent & selected feature columns
new_df = indexed.select('Stage_code', 'Stadium_code', 'City_code', 'Home_Team_code', 'Away_Team_code',
       'Referee_code', 'Assistant_1_code', 'Assistant_2_code', 'winner')
new_df.show()

display(indexed.select("Stadium_code", "winner"))


# Convert target into numerical categories
labelIndexer = StringIndexer(inputCol="winner", outputCol="label")

# VectorAssembler  to genernate a feature vector for the selected columns
assembler = VectorAssembler(
    inputCols=['Stage_code', 'Stadium_code', 'City_code', 'Home_Team_code', 'Away_Team_code',
       'Referee_code', 'Assistant_1_code', 'Assistant_2_code'],
    outputCol='features')

# Split data to test and train dataset
splits = new_df.randomSplit([0.8, 0.2], 1234)
train = splits[0]
test = splits[1]

# NaiveBayes algorithm 
nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
pipeline = Pipeline(stages=[labelIndexer, assembler, nb])
model = pipeline.fit(train)

#predict output for test data
predictions = model.transform(test)
predictions.printSchema()

predictions.select("label", "prediction", "probability").show()

#Calcuate accuracy
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                              metricName="accuracy")
evaluator.explainParam("metricName")
accuracy = evaluator.evaluate(predictions)
print( accuracy)


# RandomForestClassifier algorithm 
rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label')
pipeline = Pipeline(stages=[labelIndexer, assembler, rf])
model = pipeline.fit(train)

#predict output for test data
predictions = model.transform(test)

predictions.printSchema()
predictions.select("label", "prediction", "probability").show()

#Calcuate accuracy
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
evaluator.explainParam("metricName")
accuracy = evaluator.evaluate(predictions)
print( accuracy)


#Create new dataframe with selected column
new_df = pandas_df[['Stage_code', 'Stadium_code', 'City_code', 'Home_Team_code', 'Away_Team_code',
       'Referee_code', 'Assistant_1_code', 'Assistant_2_code']]
new_df.dtypes

#Split test and training dataset
X_train, X_test, y_train, y_test = train_test_split(new_df, pandas_df['winner'] , test_size=0.20, random_state=42)

# KNN Algorithm using Sklearn
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X_train, y_train)

#Calcuate accuracy
print(metrics.accuracy_score(y_test, neigh.predict(X_test)))

