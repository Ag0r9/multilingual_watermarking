from paths import Paths
from watermark_evaluation import WatermarkEvaluator


def main():
    # Initialize paths
    paths = Paths()
    output_dir = paths.GENERATED_TEXT_DIR / "watermark_evaluation"
    
    # Define watermark types to evaluate
    watermark_types = [
        "random_50",      # Random mask on 50% of tokens
        "adj_adv_90",     # Random mask on 90% of adjectives and adverbs
        "feminine_90",    # Random mask on 90% of words of feminine gender
        "verb_comp_90"    # Random mask on 90% of complements and verbs
    ]
    
    # Load prompts from the existing file
    from answering_output_with_logits import promts
    
    # Initialize evaluator
    evaluator = WatermarkEvaluator(
        model_name="speakleash/Bielik-7B-Instruct-v0.1",
        embedding_model_name="sentence-transformers/all-mpnet-base-v2",
        spacy_model="pl_core_news_lg"
    )
    
    # Run evaluation
    print("Starting watermark evaluation...")
    evaluator.run_evaluation(
        prompts=promts,
        watermark_types=watermark_types,
        output_dir=output_dir
    )
    print(f"Evaluation complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main() 