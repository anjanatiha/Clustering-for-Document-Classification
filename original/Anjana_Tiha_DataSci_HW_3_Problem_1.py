
# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn import mixture
from sklearn.metrics import f1_score
import time


newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')
    
def majority_assign(predictions):

    prediction_set = set(predictions)
    prediction_list = list(prediction_set)

    cluster_assign_type = {}

    for x in prediction_list:
        
        cluster_x_dict = {}
        index = 0
        
        for j in predictions:
            if x == j:
                #fraction of the data set use would help processing faster faster.
                actual_value = newsgroups_test.target[:250][index]
                #actual_value = newsgroups_test.target[index]
                if actual_value in cluster_x_dict:      
                    cluster_x_dict[actual_value] += 1
                else:
                    cluster_x_dict[actual_value] = 1
            index += 1
            
        
        for val in cluster_x_dict:
            cluster_x_majority = max (cluster_x_dict, key = cluster_x_dict.get)
        cluster_assign_type[x] = cluster_x_majority
        
        del(cluster_x_dict)
    
    assign_index = 0
    
    for f in predictions:
        
        predictions[assign_index] = cluster_assign_type[f]
        assign_index += 1
    
    return predictions


def model_performance(c_model, c_model_identifier, size, c_model_name):
    
    if c_model_identifier == 0:
        
        tfidf_vectorizer = TfidfVectorizer(max_features = 1000, min_df = 2, max_df = 0.8)
        newsgroups_train_tfidf_vectorized = tfidf_vectorizer.fit_transform(newsgroups_train.data[:1500])
        #newsgroups_train_tfidf_vectorized = tfidf_vectorizer.fit_transform(newsgroups_train.data)
        newsgroups_test_tfidf_vectorized = tfidf_vectorizer.transform(newsgroups_test.data[:250])
        #newsgroups_test_tfidf_vectorized = tfidf_vectorizer.transform(newsgroups_test.data)
        
        c_model.set_params(n_clusters = size)
        c_model.fit(newsgroups_train_tfidf_vectorized)
        predictions = c_model.predict(newsgroups_test_tfidf_vectorized)
        
        
    elif c_model_identifier == 1:
        
        tfidf_vectorizer = TfidfVectorizer(max_features = 1000, min_df = 2, max_df = 0.8)
        newsgroups_train_tfidf_vectorized = tfidf_vectorizer.fit_transform(newsgroups_train.data[:1500])
        #newsgroups_train_tfidf_vectorized = tfidf_vectorizer.fit_transform(newsgroups_train.data)
        newsgroups_test_tfidf_vectorized = tfidf_vectorizer.transform(newsgroups_test.data[:250])
        #newsgroups_test_tfidf_vectorized = tfidf_vectorizer.transform(newsgroups_test.data)
        
        c_model.set_params(n_components = size)
        c_model.fit(newsgroups_train_tfidf_vectorized.todense())
        predictions = c_model.predict(newsgroups_test_tfidf_vectorized.todense())

    predictions = majority_assign(predictions)

    f1_score_model = f1_score(newsgroups_test.target[:250], predictions, average = 'weighted')    
    #f1_score_model = f1_score(newsgroups_test.target, predictions, average = 'weighted')   
      
    print(c_model_name, size, "F-1 Score:", f1_score_model)
    

k_means1 = KMeans()
k_means2 = KMeans()
k_means3 = KMeans()
k_means4 = KMeans()
k_means5 = KMeans()
k_means6 = KMeans()
k_means7 = KMeans()
gmm1 =  mixture.GaussianMixture(covariance_type = 'diag')
gmm2 =  mixture.GaussianMixture(covariance_type = 'diag')
gmm3 =  mixture.GaussianMixture(covariance_type = 'diag')
gmm4 =  mixture.GaussianMixture(covariance_type = 'diag')
gmm5 =  mixture.GaussianMixture(covariance_type = 'diag')
gmm6 =  mixture.GaussianMixture(covariance_type = 'diag')
gmm7 =  mixture.GaussianMixture(covariance_type = 'diag')
gmm8 =  mixture.GaussianMixture(covariance_type = 'diag')

print("---------------------------------------------------------------------------------------------------------------")
print("------------------------Clustering for Document Classification Performace Evaluation---------------------------")
print("---------------------------------------------------------------------------------------------------------------")
print("------------------------------------------------KMeans---------------------------------------------------------")
model_performance(k_means1, 0, 15, "KMeans, Cluster Size =")
print("---------------------------------------------------------------------------------------------------------------")
model_performance(k_means2, 0, 21, "KMeans, Cluster Size =")
print("---------------------------------------------------------------------------------------------------------------")
model_performance(k_means3, 0, 24, "KMeans, Cluster Size =")
print("---------------------------------------------------------------------------------------------------------------")
model_performance(k_means4, 0, 26, "KMeans, Cluster Size =")
print("---------------------------------------------------------------------------------------------------------------")
model_performance(k_means5, 0, 29, "KMeans, Cluster Size =")
print("---------------------------------------------------------------------------------------------------------------")
model_performance(k_means6, 0, 31, "KMeans, Cluster Size =")
print("---------------------------------------------------------------------------------------------------------------")
model_performance(k_means7, 0, 22, "KMeans, Cluster Size =")
print("---------------------------------------------------------------------------------------------------------------")
print("-------------------------------Gaussian Mixture Model----------------------------------------------------------")
model_performance(gmm1, 1, 21, "Gaussian Mixture Model, Number of Mixture Component =")
print("---------------------------------------------------------------------------------------------------------------")
model_performance(gmm2, 1, 20, "Gaussian Mixture Model, Number of Mixture Component =")
print("---------------------------------------------------------------------------------------------------------------")
model_performance(gmm3, 1, 23, "Gaussian Mixture Model, Number of Mixture Component =")
print("---------------------------------------------------------------------------------------------------------------")
model_performance(gmm4, 1, 29, "Gaussian Mixture Model, Number of Mixture Component =")
print("---------------------------------------------------------------------------------------------------------------")
model_performance(gmm5, 1, 24, "Gaussian Mixture Model, Number of Mixture Component =")
print("---------------------------------------------------------------------------------------------------------------")
model_performance(gmm6, 1, 28, "Gaussian Mixture Model, Number of Mixture Component =")
print("---------------------------------------------------------------------------------------------------------------")
model_performance(gmm7, 1, 30, "Gaussian Mixture Model, Number of Mixture Component =")
print("---------------------------------------------------------------------------------------------------------------")
model_performance(gmm8, 1, 34, "Gaussian Mixture Model, Number of Mixture Component =")
print("---------------------------------------------------------------------------------------------------------------")


