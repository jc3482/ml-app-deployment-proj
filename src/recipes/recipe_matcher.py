class RecipeMatcher:
    """
    Simple scoring:
    score = count of overlapping ingredients
    """

    def __init__(self):
        pass

    def match(self, fridge_items, recipe_items):
        fridge_set = set(fridge_items)
        recipe_set = set(recipe_items)
        return len(fridge_set & recipe_set)
