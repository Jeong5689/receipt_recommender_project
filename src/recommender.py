import pandas as pd
import random

class Recommender:
    def __init__(self, users_csv="data/users.csv"):
        self.users = pd.read_csv(users_csv)

    def recommend(self, user_id, top_k=5):
        all_items = ["상품A", "상품B", "상품C", "상품D", "상품E"]
        return random.sample(all_items, top_k)