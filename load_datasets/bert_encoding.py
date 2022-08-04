from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
model.max_seq_length = 384


def text_emb(text):
    return model.encode(text)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Encoding inputs of several datasets')
    parser.add_argument('--type', type=str, default="onehot", metavar='N', help='dataset(onehot)')
    parser.add_argument('--data', type=str, default="imdb", metavar='N', help='dataset(imdb)')

    args = parser.parse_args()
    if args.data == "imdb":
        path = "imdb/"
        import pandas as pd

        train_data = pd.read_csv(path + 'train.csv')
        test_data = pd.read_csv(path + 'test.csv')
        train_data = train_data[:1000]
        test_data = test_data[:500]
        train_data = train_data.to_dict(orient='records')
        test_data = test_data.to_dict(orient='records')
        train_texts, train_labels = list(zip(*map(lambda d: (d['text'], d['sentiment']), train_data)))
        test_texts, test_labels = list(zip(*map(lambda d: (d['text'], d['sentiment']), test_data)))
        # for text in train_texts:
        #     print(text)
        emb_texts = []
        num_batchs = int(len(train_texts) / 4)
        for i in range(num_batchs):
            sample_texts = train_texts[i * 4:(i + 1) * 4]
            print(sample_texts)
            print(i)
            emb_texts.extend(np.asarray(text_emb(sample_texts)))

        emb_texts = np.asarray(emb_texts)
        print(emb_texts.shape)
        print(len(emb_texts))

        np.save("imdb_train_emb", emb_texts)
