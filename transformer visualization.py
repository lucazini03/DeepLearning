import gensim.downloader

model = gensim.downloader.load("glove-wiki-gigaword-50")

# The following code is used to visualize the word embeddings using PCA
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_word_embeddings(model, words):
    # Get the word vectors for the specified words
    word_vectors = [model[word] for word in words]

    # Perform PCA to reduce dimensionality to 2D
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(word_vectors)

    # Create a scatter plot of the reduced vectors
    plt.figure(figsize=(10, 10))
    plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1])

    # Annotate each point with the corresponding word
    for i, word in enumerate(words):
        plt.annotate(word, xy=(reduced_vectors[i, 0], reduced_vectors[i, 1]))

    plt.title("Word Embeddings Visualization")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid()
    plt.show()
# Example usage
words = ["tower", "cat", "dog", "car", "tree", "castle", "ferrari", "computer", "apple", "banana"]
plot_word_embeddings(model, words)