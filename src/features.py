from sklearn.feature_extraction.text import TfidfVectorizer


def create_tfidf_vectorizer() -> TfidfVectorizer:
    """
    TF-IDF configuration optimized for a fast CPU text-classification model.

    - unigrams + bigrams capture both individual terms and phrases
    - min_df reduces extremely rare noisy features
    - max_df removes terms occurring in nearly every complaint
    - sublinear_tf reduces the influence of repeated terms
    """
    return TfidfVectorizer(
        lowercase=False,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.98,
        sublinear_tf=True,
        max_features=100_000,
    )
