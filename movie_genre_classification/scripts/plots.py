import matplotlib.pyplot as plt
import seaborn as sns

def genre_distibution(train_data):
    plt.figure(figsize=(10, 6))
    train_data['genre'].explode().value_counts().plot(kind='bar')

    plt.xticks(rotation=45)
    plt.xlabel('Genres')
    plt.ylabel('Frequency')
    plt.title('Genre Distribution in Training Data')
    plt.tight_layout() 
    plt.show()


def movie_plot_distribution(train_data):
    train_data['plot_length'] = train_data['plot'].apply(len)

    plt.figure(figsize=(12, 6))
    sns.histplot(train_data['plot_length'], bins=50, kde=True)
    plt.xlabel('Length of Plot Summary')
    plt.ylabel('Frequency')
    plt.title('Distribution of Plot Summary Lengths')
    plt.show()


def top_5_genres(train_data):
    top_genres = train_data['genre'].explode().value_counts().head(5)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_genres.index, y=top_genres.values)
    plt.xticks(rotation=45)
    plt.xlabel('Genres')
    plt.ylabel('Frequency')
    plt.title('Top 10 Genres')
    plt.show()