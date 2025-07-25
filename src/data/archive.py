
class Archive:
    def __init__(self, type, labeled_bow=None, reviews=None):
        self.type = type
        self.labeled_bow = labeled_bow
        self.reviews = reviews if reviews is not None else []
        self.size = len(self.reviews)

    def add_review(self, review):
        self.reviews.append(review)
        self.size += 1
    
    def add_labeled_bow(self, labeled_bow):
        self.labeled_bow = labeled_bow
    
    def to_dict(self):
        return {
            "type": self.type,
            "labeled_bow": self.labeled_bow,
            "reviews": self.reviews,
            "size": self.size,
        }