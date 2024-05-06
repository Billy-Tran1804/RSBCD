# RSBCD
Graph-Based Recommendation Systems in Social Network Analysis Based on Community Detection Algorithms
This project develops a recommendation systems based on different community detection-based algorithms and compare
them with a graph neural network-based (GNN) recommendation system using various metrics like normalized discounted
cumulative gain, precision and recall, and time complexity. We implement the different community detection algorithms with
real social network datasets. In the next step, we use outputs from the previous step to develop our proposed community-based
recommendation systems. On the other hand, we reproduce a state-of-the-art GNN-based recommendation system. Finally,
we measure and compare our proposed recommendation system with the GNN recommedation system in using standard
various metrics. Although, the result show that our proposed recommendation system does not have a better performance in
comparison with GNN, but it also encourage us to research more in this direction in the future
1 INTRODUCTION
Recommendation systems are crucial for boosting user engagement, enhancing content discovery, and elevating
both content consumption and user satisfaction [ 5 ]. The critical aspect of recommendation systems in social
networks is the analysis of the extensive and intricate network of users, content, and their interactions. It
also includes examining how interactions between users and user-generated content evolve over time. There-
fore, recommendation systems must also adapt and evolve in response to these changes[14]. Various types of
recommendation systems have been developed, including content-based filtering, collaborative filtering, and
hybrid filtering approaches [ 9 ]. Content-based filtering, a highly effective method of recommendation, depends
on the relationships between content elements. It suggests items to users by analyzing item descriptions and
considering user preferences [ 12]. In collaborative filtering, recommendations are made to users based on their
own behavior and the behavior of others who are similar to them. This approach uses datasets of user preferences
to create personalized suggestions[ 4 ]. According to Ricci, collaborative filtering algorithms are primarily divided
into three categories: memory-based, model-based, and hybrid approaches. [ 11 ]. The hybrid filtering technique
combines various recommendation methods to enhance system performance. It addresses the limitations of both
content-based and collaborative filtering techniques by capitalizing on their complementary strengths.[4].
Among the three types of collaborative filtering recommendation systems, we opted to develop a model-based
collaborative filtering system. Model-based filtering approaches create models using subsets of datasets, leverag-
ing historical user ratings to improve the performance of collaborative filtering. There are numerous algorithms
and techniques available for constructing a model-based recommendation system, including the use of Clustering
