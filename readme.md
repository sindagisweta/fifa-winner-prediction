# Football Winner predictaion based on FIFA historic data

Prediction of Football winner using machine algorithms CNN, Naive Bayes, Random Forest. 
This project uses hostoric FIFA world cup data to train the model

## Pre-requesties
1. Hadooop instance with Spark installed
2. Python IDE
3. Install below listed python library

## Python Librarys to be installed	

- matplotlib
- sklearn
- pandas
- numpy

## Steps to run the code 
1. Upload the dataset to HDFS using the below command
```
	hdfs dfs -copyFromLocal /data/WorldCupMatches.csv
	hdfs dfs -copyFromLocal /data/Result.csv
```
2. Run the pyspark file by specifiying the host edge node

Note : You can run the code line by line using Pyspark REBL or submit the whole code as job. Submitting the code as job wont let you view the exploratory data analysis part.
