import numpy as np
import pandas as pd
import pickle
import matrix_factorization_utilities
import os
import webbrowser

# Load user ratings
raw_training_dataset_df = pd.read_csv('TrainingData/movie_ratings_data_set_training.csv')
# raw_training_dataset_df = pd.read_csv('TrainingData/MR.csv')

# raw_df = pd.read_csv('TrainingData/MR.csv')
# raw_training_dataset_df = raw_df.head(1000)

# print('hi0')
# Convert the running list of user ratings into a matrix
ratings_training_df = pd.pivot_table(raw_training_dataset_df, index='user_id', columns='movie_id', aggfunc=np.max)
# print('hi1')
# Normalize the ratings (center them around their mean)
normalized_ratings, means = matrix_factorization_utilities.normalize_ratings(ratings_training_df.as_matrix())
# print('hi2')
# Apply matrix factorization to find the latent features
U, M = matrix_factorization_utilities.low_rank_matrix_factorization(normalized_ratings,
                                                                    num_features=12,
                                                                    regularization_amount=1)
# print('hi3')

# Find all predicted ratings by multiplying U and M
predicted_ratings = np.matmul(U, M)

# Add back in the mean ratings for each product to de-normalize the predicted results
predicted_ratings = predicted_ratings + means


# Measure RMSE
rmse_training = matrix_factorization_utilities.RMSE(ratings_training_df.as_matrix(),
                                                    predicted_ratings)
# rmse_testing = matrix_factorization_utilities.RMSE(ratings_testing_df.as_matrix(),
#                                                    predicted_ratings)

print("Training RMSE: {}".format(rmse_training))
# print("Testing RMSE: {}".format(rmse_testing))

# Save features and predicted ratings to files for later use
pickle.dump(U, open("../Recommendation/TrainingResultsData/user_features.dat", "wb"))
pickle.dump(M, open("../Recommendation/TrainingResultsData/product_features.dat", "wb"))
pickle.dump(predicted_ratings, open("../Recommendation/TrainingResultsData/predicted_ratings.dat", "wb" ))
pickle.dump(means, open("../Recommendation/TrainingResultsData/means.dat", "wb" ))