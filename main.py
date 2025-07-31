from src.data.processor import unzip_data_extract_contents
from src.model.train_model import train_bow_logreg, train_tfidf_logreg, finetune_bert
from src.model.predict_model import predict_sentiment_tfidf, predict_sentiment_bow, predict_sentiment_bert

def main():
    test_archive, train_archive, unsup_archive, imdb_vocab, imdb_expected_rating = unzip_data_extract_contents()
    train_tfidf_logreg(test_archive, train_archive, imdb_vocab, imdb_expected_rating)
    train_bow_logreg(test_archive, train_archive, imdb_vocab)
    finetune_bert(train_archive, test_archive)

    reviews = [
        "this movie was great and the acting was amazing",
        "this movie was bad and the acting was terrible",]
    for review in reviews:
        print("TFIDF Prediction:", predict_sentiment_tfidf(review))
        print("BoW Prediction:", predict_sentiment_bow(review, imdb_vocab))
        print("BERT Prediction:", predict_sentiment_bert(review))

        print('With best params:')
        print("TFIDF Prediction:", predict_sentiment_tfidf(review, with_params=True))
        print("BoW Prediction:", predict_sentiment_bow(review, imdb_vocab, with_params=True))

main()

