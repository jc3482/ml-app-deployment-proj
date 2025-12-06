"""
Main entry point for SmartPantry Recipe Matching System.
CLI interface for data processing and recipe matching.
"""
import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from recipe_matcher.pipeline import (
    normalize_recipe_dataset,
    apply_ontology_processing,
    RecipePipeline
)


def main():
    parser = argparse.ArgumentParser(
        description="SmartPantry: Recipe Matching System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # =========================================================================
    # Command 1: normalize
    # =========================================================================
    normalize_parser = subparsers.add_parser(
        'normalize',
        help='Step 1: Normalize raw recipe dataset'
    )
    normalize_parser.add_argument(
        '--input', type=str,
        help='Path to raw recipe CSV'
    )
    normalize_parser.add_argument(
        '--no-json', action='store_true',
        help='Skip JSON output'
    )
    normalize_parser.add_argument(
        '--no-csv', action='store_true',
        help='Skip CSV output'
    )
    
    # =========================================================================
    # Command 2: ontology
    # =========================================================================
    ontology_parser = subparsers.add_parser(
        'ontology',
        help='Step 2: Apply ontology processing to normalized dataset'
    )
    ontology_parser.add_argument(
        '--input', type=str,
        help='Path to normalized recipes'
    )
    ontology_parser.add_argument(
        '--no-json', action='store_true',
        help='Skip JSON output'
    )
    ontology_parser.add_argument(
        '--no-csv', action='store_true',
        help='Skip CSV output'
    )
    # add toggle for .pkl output (cached_ontology.pkl)
    ontology_parser.add_argument(
        '--no-pkl', action='store_true',
        help='Skip PKL output (cached_ontology.pkl)'
    )
    
    # =========================================================================
    # Command 3: match
    # =========================================================================
    match_parser = subparsers.add_parser(
        'match',
        help='Match ingredients against recipe database'
    )
    match_parser.add_argument(
        '--ingredients', type=str,
        help='Comma-separated ingredients'
    )
    match_parser.add_argument(
        '--input', type=str,
        help='Path to JSON file with ingredients'
    )
    match_parser.add_argument(
        '--recipes', type=str,
        help='Path to recipe dataset'
    )
    match_parser.add_argument(
        '--topk', type=int, default=5,
        help='Number of top recipes to show'
    )
    match_parser.add_argument(
        '--no-ontology', action='store_true',
        help='Use normalized data only (skip ontology)'
    )
    
    args = parser.parse_args()
    
    # =========================================================================
    # Execute Commands
    # =========================================================================
    
    if args.command == 'normalize':
        normalize_recipe_dataset(
            input_path=args.input,
            output_json=not args.no_json,
            output_csv=not args.no_csv
        )
    
    elif args.command == 'ontology':
        apply_ontology_processing(
            input_path=args.input,
            output_json=not args.no_json,
            output_csv=not args.no_csv,
            output_pkl=not args.no_pkl
        )
    
    elif args.command == 'match':
        # Load user ingredients
        if args.ingredients:
            user_ingredients = [x.strip() for x in args.ingredients.split(",")]
        elif args.input:
            with open(args.input, 'r') as f:
                data = json.load(f)
                user_ingredients = data if isinstance(data, list) else data.get('ingredients', [])
        else:
            print("Error: Must provide --ingredients or --input")
            return 1
        
        print("\n" + "="*70)
        print("SMARTPANTRY RECIPE MATCHING")
        print("="*70)
        print(f"Detected Ingredients: {user_ingredients}")
        
        # Initialize pipeline
        use_ontology = not args.no_ontology
        pipeline = RecipePipeline(use_ontology=use_ontology, recipe_path=args.recipes)
        
        # Run matching
        user_processed, top_recipes = pipeline.run(user_ingredients, top_k=args.topk)
        
        # Display results
        print("\n" + "="*70)
        print(f"TOP {args.topk} RECIPE RECOMMENDATIONS")
        print("="*70)
        
        for i, recipe in enumerate(top_recipes, start=1):
            print(f"\n{'-'*70}")
            print(f"#{i}. {recipe['title']}")
            print(f"{'-'*70}")
            print(f"   Fuzzy Match Score: {recipe['fuzzy_score']:.2%}")
            print(f"   Exact Ingredient Overlap: {recipe['overlap']}")
            
            # Matched ingredients
            print(f"\n   Matched Ingredients ({len(recipe['matched'])}):")
            if recipe['matched']:
                for ing in recipe['matched'][:5]:
                    print(f"      - {ing}")
                if len(recipe['matched']) > 5:
                    print(f"      ... and {len(recipe['matched']) - 5} more")
            else:
                print("      (None)")
            
            # Missing ingredients
            print(f"\n   Missing Ingredients ({len(recipe['missing'])}):")
            if recipe['missing']:
                for ing in recipe['missing'][:5]:
                    print(f"      - {ing}")
                if len(recipe['missing']) > 5:
                    print(f"      ... and {len(recipe['missing']) - 5} more")
            else:
                print("      You have everything!")
            
            # Fuzzy matched pairs
            if recipe.get('fuzzy_pairs'):
                print(f"\n   Fuzzy Matches:")
                for pair in recipe['fuzzy_pairs'][:3]:
                    print(f"      - {pair['user']} <-> {pair['recipe']} ({pair['score']:.0%})")
            
            # Image path
            if recipe.get('image_path'):
                print(f"\n   Image: {recipe['image_path']}")
            elif recipe.get('image_name'):
                print(f"\n   Image: data/food_images/{recipe['image_name']}.jpg")
            
            # Cooking steps preview
            if recipe.get('instructions'):
                steps = recipe['instructions'][:150].replace('\n', ' ')
                print(f"\n   Steps: {steps}...")
        
        print("\n" + "="*70)
        print("Done. Happy Cooking!")
        print("="*70)
    
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
