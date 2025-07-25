from src.data.processor import unzip_data_extract_contents
from src.model.train_model import train_bow_logreg, train_tfidf_logreg
from src.model.predict_model import predict_sentiment_tfidf, predict_sentiment_bow

def main():
    test_archive, train_archive, unsup_archive, imdb_vocab, imdb_expected_rating = unzip_data_extract_contents()
    # train_tfidf_logreg(test_archive, train_archive, imdb_vocab, imdb_expected_rating)
    # train_bow_logreg(test_archive, train_archive, imdb_vocab)

    sample_text = "this movie was mid"
    print("TFIDF Prediction:", predict_sentiment_tfidf(sample_text))
    print("BoW Prediction:", predict_sentiment_bow(sample_text, imdb_vocab))

    sample_text = "it wasnt bad"
    print("TFIDF Prediction:", predict_sentiment_tfidf(sample_text))
    print("BoW Prediction:", predict_sentiment_bow(sample_text, imdb_vocab))


main()

