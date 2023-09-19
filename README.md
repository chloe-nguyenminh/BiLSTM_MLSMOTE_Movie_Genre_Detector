# Movie_Genre_Detector_Project
`src/text-classifier-bisltm`: Movie Genre Detector given the title and description using multi-label text classification and standard natural language processing methods.
Implemented using Bi-Directional Long short-term memory networks (BiLSTM), a variety of recurrent neural networks (RNNs). 

`src/image-classifier-mlsmote`: Movie Genre Detector given poster images, using multi-label image classification. 
Implemented using the Multi-Label Synthetic Minority Over-sampling Technique. 
ALgorithm adapted from https://doi.org/10.1613/jair.953.

Classification is a supervised learning technique that deals with the categorization of a data object into one of the several predefined classes. Majority of the methods for supervised machine learning proceeds from a formal setting in which the data objects (instances) are represented in the form of feature vectors wherein each object is associated with a unique class label from a set of disjoint class labels. 

In this particular problem with the Movie Genre Detector, we are concerned with **Multi-label Classification**, in which the target variable has more than one dimension where each dimension is binary (i.e., contain only two distinct values). For instance, a movie can be associated with more than one genres (e.g., Drama and Comedy); however, a movie can only be a genre (1) or not (0) (e.g., either is a Comedy or not, cannot be half Comedy).

In some of the classification cases, the number of instances associated with one class is way lesser than the other clas. This leads to the problem of data imbalance, which greatly affects our machine learning algorithm performance. This problem also arises in the case of multi-label classification where the labels are unevenly distributed. Please refer to `notebooks/organize_data.ipynb` for demonstration of the dataset skewedness. Here, I adapted a data augmentation method for imbalance multi-label data, known as Multi-Label Synthetic Minority Over-Sampling (MLSMOTE).

The algorithm is an extension of the original SMOTE algorithm, first proposed by Chawla, Bowyer, Hall and Kegelmeyer in 2002 (https://doi.org/10.1613/jair.953). 


