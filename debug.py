import pickle

cache = pickle.load(open("data/normalized_recipes.pkl", "rb"))

count_empty = sum(1 for k,v in cache.items() if len(v["normalized"]) == 0)
count_total = len(cache)

print("total recipes:", count_total)
print("recipes with empty normalized list:", count_empty)

example = list(cache.items())[0]
print("sample recipe:\n", example)
