import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import col, when
import os

# Get current directory (folder containing app.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Paths to datasets (relative to the app.py file)
ratings_path = os.path.join(current_dir, "ratings.csv")
movies_path = os.path.join(current_dir, "movies.csv")

# Start Spark session
spark = SparkSession.builder.appName("Movie Recommender").getOrCreate()

# Load datasets
ratings = spark.read.csv(ratings_path, header=True, inferSchema=True).dropna()
movies = spark.read.csv(movies_path, header=True, inferSchema=True).dropna()

# Streamlit UI
st.title("🎬 Movie Recommendation System")
user_input = st.text_input("Enter a User ID to get movie recommendations:", "25")

if user_input:
    try:
        user_id = int(user_input)

        # Train ALS model
        als = ALS(
            userCol="userId",
            itemCol="movieId",
            ratingCol="rating",
            coldStartStrategy="drop",
            nonnegative=True,
            implicitPrefs=False
        )
        model = als.fit(ratings)

        # Find movies not yet rated by this user
        rated_movie_ids = ratings.filter(ratings.userId == user_id).select("movieId")
        unrated_movies = movies.select("movieId").distinct().join(rated_movie_ids, on="movieId", how="left_anti")
        user_unrated = unrated_movies.withColumn("userId", col("movieId") * 0 + user_id)

        # Predict ratings
        predictions = model.transform(user_unrated)

        # Clip predictions to 0-5 range
        predictions = predictions.withColumn("prediction", when(col("prediction") > 5.0, 5.0)
                                                         .when(col("prediction") < 0.0, 0.0)
                                                         .otherwise(col("prediction")))

        # Join with movie titles and sort
        recommended = predictions.join(movies, on="movieId") \
                                 .select("title", "prediction") \
                                 .orderBy(col("prediction").desc()) \
                                 .limit(5)

        st.subheader(f"Top 5 Recommendations for User ID: {user_id}")
        st.dataframe(recommended.toPandas())

    except ValueError:
        st.error("❌ Please enter a valid numeric User ID.")
