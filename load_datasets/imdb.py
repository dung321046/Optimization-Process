from keras.datasets import imdb

# Load the data encoded by 10,000 of the most frequently occurring words
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

if __name__ == "__main__":
    review_lens = [len(sequence) for sequence in train_data]
    from visualization.histogram import histogram_uni

    print(max(review_lens))
    histogram_uni(review_lens, bins=20, title="The distributions of word counts in imdb reviews")


    def reverse_to_string(seq):
        word_index = imdb.get_word_index()
        reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
        # indices are off by 3 because 0, 1, and 2 are reserverd indices for "padding", "Start of sequence" and "unknown"
        return ' '.join([reverse_word_index.get(i - 3, '?') for i in seq])


    print(reverse_to_string(train_data[0]))
