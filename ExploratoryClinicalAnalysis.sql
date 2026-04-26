-- Databricks notebook source
-- MAGIC %md
-- MAGIC # INTRODUCTION
-- MAGIC In this task, I’m analysing a large clinical trials dataset using Spark SQL within Databricks. The dataset is quite substantial, with over 570,000 records, where each row represents a clinical study carried out in the United States. It includes useful details such as the type of study, its current status, the medical conditions being researched, and key dates like when the study started and ended. The main goal here is to show how Spark SQL can be used to handle and analyse large-scale data efficiently. Because the dataset is so big, using standard tools would be slow and impractical, so Spark provides a much more scalable way of working with it. To keep things organised, the analysis is broken down into a few key parts. First, I look at the different types of clinical trials and how common each one is. Then, I identify the most frequently studied medical conditions. After that, I calculate the average duration of clinical trials in months. Finally, I explore how the number of Alzheimer’s-related studies has changed over time. Overall, this task highlights how large healthcare datasets can be analysed to uncover useful insights and trends.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC # READING THE DATASET FROM UNITY CATALOG
-- MAGIC I read the CSV dataset from Unity Catalog into a Spark DataFrame called df. I used spark.read and set header=True so that the first row is treated as column names instead of actual data. Since the dataset may contain fields with line breaks, I enabled multiLine=True to make sure those values are read correctly. I also set the escape option to a double quote so that any quotation marks within the data don’t cause parsing issues. To avoid manually defining data types, I used inferSchema=True, which allows Spark to automatically detect the correct data type for each column. I then loaded the file using the .csv() method by specifying the path in Unity Catalog. Finally, I used display(df) to visualise the DataFrame in a table format within Databricks so I can easily inspect the data.

-- COMMAND ----------

-- MAGIC %python
-- MAGIC #Reading the dataset from Unity Catalog into a variable titled df
-- MAGIC df = (
-- MAGIC     spark.read
-- MAGIC     .option("header", True)
-- MAGIC     .option("multiLine", True)     # emails may contain line breaks
-- MAGIC     .option("escape", '"')
-- MAGIC     .option("inferSchema", True)
-- MAGIC     .csv("/Volumes/teaching/datasets/assignment_2/task_1/ctg-studies.csv")
-- MAGIC )
-- MAGIC
-- MAGIC display(df)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## ANALYSIS OF DIFFERENT TYPES OF CLINICAL TRIALS IN THE DATASET
-- MAGIC The next step I took was to analyse the different types of clinical trials in the dataset. I started by importing the count and col functions from PySpark, which I used for aggregation and column referencing. I then filtered the DataFrame to remove any rows where the Study Type column is either null or an empty string, so that only valid values are included in the analysis. After cleaning the data, I grouped the remaining records by Study Type to see how many entries fall under each category. I used the count("*") function to count the total number of records in each group and gave this new column a clearer name, Clinical Trial categories. I then sorted the results in descending order so that the most common study types appear at the top. Finally, I used display(trial_types) to present the results in a table format within Databricks, making it easy to view and interpret the distribution of clinical trial types.

-- COMMAND ----------

-- MAGIC %python
-- MAGIC from pyspark.sql.functions import count, col
-- MAGIC
-- MAGIC trial_types = (
-- MAGIC     df
-- MAGIC     .filter(col("Study Type").isNotNull() & (col("Study Type") != ""))   #removing missing senders and empty strings
-- MAGIC     .groupBy("Study Type")
-- MAGIC     .agg(count("*").alias("Clinical Trial categories"))
-- MAGIC     .orderBy(col("Clinical Trial categories").desc())
-- MAGIC )
-- MAGIC
-- MAGIC display(trial_types)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## TOP 10 CONDITIONS
-- MAGIC ### SEPERATING DIFFERENT MEDICAL CONDITIONS TO ALLOW BETTER ANALYSIS
-- MAGIC In this step, I transformed the Conditions column so that each condition is stored separately rather than as a single combined string. I started by importing the split, explode, and col functions from PySpark. I then used withColumn to overwrite the existing Conditions column. First, I applied the split function to break the string into an array using the pipe symbol (|) as the delimiter. After that, I used the explode function to turn each element of that array into its own row, meaning that each condition now appears on a separate row instead of being grouped together. This makes the data easier to analyse, especially when counting or filtering specific conditions. Finally, I used display(df) to view the updated DataFrame in Databricks.

-- COMMAND ----------

-- MAGIC %python
-- MAGIC from pyspark.sql.functions import split, explode, col
-- MAGIC
-- MAGIC df = df.withColumn(
-- MAGIC     "Conditions",
-- MAGIC     explode(split(col("Conditions"), "\\|"))
-- MAGIC )
-- MAGIC
-- MAGIC display(df)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## IDENTIFYING AND DISPLAYING THE TOP 10 MEDICAL CONDITIONS BASED ON CLINICAL TRIALS
-- MAGIC I started by importing the count function from PySpark so I could perform aggregation. I then filtered the DataFrame to remove any rows where the Conditions column is null or empty, ensuring that only valid condition values are included in the analysis. After cleaning the data, I grouped the records by Conditions to count how many trials are associated with each condition. I used count("*") to calculate the total number of trials for each group and renamed this column to "Number of Trials" for clarity. I then sorted the results in descending order so that the most common conditions appear first. I then limited the output to the top 10 conditions as requested. Finally, I used display(top_conditions) to present the results in a table format within Databricks, making it easy to interpret which conditions have the highest number of clinical trials.
-- MAGIC

-- COMMAND ----------

-- MAGIC %python
-- MAGIC #collating the top 10 Medical conditions by number of clinical trials from the exploded Conditions column
-- MAGIC from pyspark.sql.functions import count
-- MAGIC
-- MAGIC top_conditions = (
-- MAGIC     df
-- MAGIC     .filter(col("Conditions").isNotNull() & (col("Conditions") != ""))   #removing missing senders and empty strings
-- MAGIC     .groupBy("Conditions")
-- MAGIC     .agg(count("*").alias("Number of Trials"))
-- MAGIC     .orderBy(col("Number of Trials").desc())
-- MAGIC     .limit(10)
-- MAGIC )
-- MAGIC
-- MAGIC display(top_conditions)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC # THE MEAN TRIAL LENGTH IN MONTHS
-- MAGIC The next thing I did was to calculate the average duration of clinical trials in months. I started by filtering the DataFrame to include only rows where the Completion date is not null, since I need both a start and end date to measure the trial length. I then used the months_between function to calculate the difference in months between the Completion date and the Start date for each trial. After that, I applied the avg function to compute the mean of these durations across all valid records, and I gave the result a clear name, Mean trial length in Months. Finally, I used .show() to display the result directly in the Databricks output.

-- COMMAND ----------

-- MAGIC %python
-- MAGIC from pyspark.sql.functions import col, months_between, avg
-- MAGIC
-- MAGIC df.filter(col("Completion date").isNotNull()) \
-- MAGIC   .select(avg(months_between(col("Completion date"), col("Start date"))).alias("Mean trial length in Months")) \
-- MAGIC   .show()

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## ANALYSIS OF THE NUMBER OF ALZHEIMER'S RELATED CLINICAL TRIALS CARRIED OUT FROM 1996 TO 2010 AND THE TREND OVER THAT PERIOD
-- MAGIC In this step, I analysed how the number of completed Alzheimer’s-related clinical trials has changed over time. I started by filtering the DataFrame to include only records where the Completion date is not null, the Study Status is marked as COMPLETED, and the Conditions column contains the term “alzheimer”. I used the lower function to make the condition check case-insensitive, ensuring that variations like “Alzheimer” or “alzheimer” are all captured. After filtering the data, I created a new column called year by extracting the year from the Completion date using the year function. This allows me to analyse trends over time. I then grouped the data by this year column and used count() to calculate how many trials were completed each year. Finally, I sorted the results in ascending order of year so the trend appears in chronological order, and displayed the output using display(df_counts). To visualise this trend in Databricks, I used the built-in plotting feature directly from the displayed table. After running display(df_counts), I switched to the chart view and selected a line chart. I set the year column as the x-axis and the count column as the y-axis. This created a clear line graph showing how the number of completed Alzheimer’s trials changed over time, making it easier to identify patterns or trends.

-- COMMAND ----------

-- MAGIC %python
-- MAGIC from pyspark.sql.functions import col, lower, year
-- MAGIC
-- MAGIC df_counts = df.filter(
-- MAGIC         (col("Completion date").isNotNull()) &
-- MAGIC         (col("Study Status") == "COMPLETED") &
-- MAGIC         (lower(col("Conditions")).contains("alzheimer"))
-- MAGIC     ) \
-- MAGIC     .withColumn("year", year(col("Completion date"))) \
-- MAGIC     .groupBy("year") \
-- MAGIC     .count() \
-- MAGIC     .orderBy("year")
-- MAGIC
-- MAGIC display(df_counts)

-- COMMAND ----------

-- DBTITLE 1,Conclusion
-- MAGIC %md
-- MAGIC ## CONCLUSION
-- MAGIC This analysis demonstrated how Spark SQL within Databricks can be used to efficiently process and extract meaningful insights from a large-scale clinical trials dataset containing over 570,000 records. By leveraging PySpark's distributed computing capabilities, the analysis was carried out quickly and at scale, which would not have been practical using traditional tools.
-- MAGIC
-- MAGIC The key findings from this analysis are as follows:
-- MAGIC - **Clinical Trial Types**: The dataset is heavily dominated by interventional studies (437,333), followed by observational studies (133,604), with expanded access trials making up a very small proportion (1,033). This reflects the research community's strong focus on testing new treatments and interventions.
-- MAGIC - **Top Medical Conditions**: The most frequently studied condition is "Healthy" (10,873 trials), which typically refers to studies involving healthy volunteers for baseline comparisons. Breast Cancer ranks second with 8,511 trials, followed by Obesity (7,324), Stroke (5,034), and Hypertension (4,510), highlighting significant research investment in chronic and life-threatening conditions.
-- MAGIC - **Average Trial Duration**: The mean clinical trial length was calculated at approximately 39.2 months (just over 3 years), indicating that clinical research is a lengthy process requiring sustained commitment and funding.
-- MAGIC - **Alzheimer's Research Trends**: The trend analysis revealed a clear and consistent increase in completed Alzheimer's-related clinical trials over time, growing from just 1 study in 1996 to 158 in 2024. This upward trend reflects growing global awareness, increased funding, and the urgent need for effective treatments as Alzheimer's prevalence continues to rise worldwide.
-- MAGIC
-- MAGIC Overall, this task has shown how big data tools like Spark SQL can be applied to healthcare datasets to uncover valuable trends and patterns that support data-driven decision-making in medical research.
-- MAGIC