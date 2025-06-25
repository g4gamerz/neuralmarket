
import json
import os 
import logging
from transformers import AutoModelForSequenceClassification
import nltk
import datetime
import pandas as pd
import sys 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "model")
STOCK_MAPPING_FILE = os.path.join(SCRIPT_DIR, "names.json")

try:
    from .gnews import GNews 
except ImportError:
    try:
        from gnews import GNews 
    except ImportError:
        print("ERROR: Could not import GNews. Ensure gnews.py is in the analysis_api directory or 'gnewsclient' is installed.")
        class GNews: 
            def __init__(self, language="en", country="US", max_results=100, period=None, start_date=None, end_date=None, exclude_websites=None, proxy=None):
                print(f"WARNING: Using placeholder GNews class. Start_date: {start_date}")
            def get_news(self, query): 
                print(f"Placeholder: Would search for {query}")
                return []

try:
    from .finbert.finbert import predict as finbert_predict 
except ImportError as e:
    logger_for_import_error = logging.getLogger(__name__ + "_import_finbert") 
    logger_for_import_error.error(f"Initial ERROR trying `from .finbert.finbert import predict`: {e}")
    logger_for_import_error.error("This usually means 'analysis_api/finbert/' is not found or not a package (missing __init__.py), OR that the 'finbert.py' module within 'analysis_api/finbert/' has an internal error (like its own imports).")
    try:
        logger_for_import_error.info("Attempting fallback import: `from .finbert import predict` (assumes analysis_api/finbert.py)")
        from .finbert.finbert import predict as finbert_predict
        logger_for_import_error.info("Fallback import `from .finbert import predict` SUCCEEDED.")
    except ImportError as e2:
        logger_for_import_error.error(f"Fallback ERROR trying `from .finbert import predict`: {e2}")
        logger_for_import_error.error("This means 'analysis_api/finbert.py' was not found or 'predict' is not in it.")
        logger_for_import_error.error("CRITICAL CHECK: If your 'finbert.py' (the one with the predict function) is located at 'analysis_api/finbert/finbert.py' AND it contains a line like 'from finbert.utils import ...', that line MUST be changed to 'from .utils import ...' (note the leading dot) for relative imports within its own package to work. Ensure 'utils.py' is in 'analysis_api/finbert/'. Also ensure 'analysis_api/finbert/__init__.py' exists.")
        logger_for_import_error.error("Sentiment analysis will use placeholder.")
        def finbert_predict(text, model, write_to_csv=False, path=None, use_gpu=False, gpu_name='cuda:0', batch_size=5):
            print("WARNING: Using placeholder FinBERT predict function. Returning empty DataFrame.")
            return pd.DataFrame(columns=['sentence', 'logit', 'prediction', 'sentiment_score'])

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_psx_stock_mapping(filepath):
    symbol_to_name = {}
    name_to_symbol = {}
    symbol_to_target_terms = {} 
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            logger.error(f"Error: {filepath} is not a JSON list as expected. Found type: {type(data)}")
            return {}, {}, {}
        for stock_info in data:
            if not isinstance(stock_info, dict):
                logger.warning(f"Skipping non-dictionary item in {filepath}: {stock_info}")
                continue
            symbol = stock_info.get('Code') 
            name = stock_info.get('Name')  
            if symbol and name:
                symbol = symbol.strip().upper()
                name_original = name.strip()
                name_lower = name_original.lower()
                symbol_to_name[symbol] = name_original
                name_to_symbol[name_lower] = symbol
                terms = {name_lower, symbol.lower()}
                name_parts_to_add = name_lower
                for suffix in [' limited', ' ltd', ' corporation', ' group', ' holdings', ' (pakistan)', ' pakistan']:
                    name_parts_to_add = name_parts_to_add.replace(suffix, '')
                name_parts = [part for part in name_parts_to_add.replace('.', '').split() if len(part) > 2]
                terms.update(name_parts)
                symbol_to_target_terms[symbol] = list(set(terms))
            else:
                logger.warning(f"Skipping stock entry due to missing 'Code' or 'Name': {stock_info}")
        logger.info(f"Successfully loaded {len(symbol_to_name)} stock mappings from {filepath}.")
        return symbol_to_name, name_to_symbol, symbol_to_target_terms
    except FileNotFoundError: logger.error(f"Stock mapping file '{filepath}' not found.")
    except json.JSONDecodeError: logger.error(f"Error decoding JSON from '{filepath}'.")
    except Exception as e: logger.error(f"Error loading stock mapping from '{filepath}': {e}", exc_info=True)
    return {}, {}, {}

PSX_SYMBOL_TO_NAME, PSX_NAME_TO_SYMBOL, PSX_SYMBOL_TO_TARGET_TERMS = load_psx_stock_mapping(STOCK_MAPPING_FILE)

def scrape_news_for_stock(stock_query, country='PK', language='en', articles_to_fetch=20, start_date_tuple=(2025,1,1)):
    logger.info(f"Scraping news for: '{stock_query}' from date {start_date_tuple}, aiming for {articles_to_fetch} articles.")
    try:
        google_news = GNews(language=language, country=country, start_date=start_date_tuple, max_results=articles_to_fetch)
        news_items = google_news.get_news(stock_query)
        logger.info(f"Found {len(news_items)} news items for '{stock_query}'.")
        return news_items
    except Exception as e:
        logger.error(f"Error during news scraping for '{stock_query}': {e}")
        return []

def analyze_sentiment_for_texts(texts_to_analyze, model_path_abs): 
    if not texts_to_analyze:
        logger.info("No texts provided for sentiment analysis.")
        return [] 
    logger.info(f"Loading FinBERT model from absolute path: {model_path_abs}")
    try:
        try: nltk.data.find('tokenizers/punkt')
        except LookupError: nltk.download('punkt', quiet=True); logger.info("NLTK 'punkt' downloaded.")
        else: logger.info("NLTK 'punkt' resource found.")
        
        model = AutoModelForSequenceClassification.from_pretrained(model_path_abs, num_labels=3, cache_dir=None)
        logger.info("FinBERT model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading FinBERT model from '{model_path_abs}': {e}", exc_info=True)
        return [] 
        
    logger.info(f"Analyzing sentiment for {len(texts_to_analyze)} texts using finbert's predict...")
    all_results = [] 
    try:
        for i, single_text in enumerate(texts_to_analyze):
            if not single_text.strip(): 
                logger.warning(f"Skipping empty text at index {i}.")
                all_results.append({'original_input_text': single_text, 
                                     'sentence_analyzed': '', 
                                     'sentiment_label': 'N/A', 
                                     'sentiment_score': 0.0})
                continue
            
            logger.info(f"Input text for FinBERT [{i}]: '{single_text}'") 
            prediction_df_for_single_text = finbert_predict(single_text, model, write_to_csv=False) 
            
            if prediction_df_for_single_text is not None and not prediction_df_for_single_text.empty:
                logger.info(f"FinBERT output for text [{i}] (DataFrame head):\n{prediction_df_for_single_text.head().to_string()}")
                for _, row in prediction_df_for_single_text.iterrows(): 
                    all_results.append({
                        'original_input_text': single_text, 
                        'sentence_analyzed': row.get('sentence', ''), 
                        'sentiment_label': str(row.get('prediction', 'Neutral')).capitalize(),
                        'sentiment_score': float(row.get('sentiment_score', 0.0))
                    })
            else:
                 logger.warning(f"FinBERT returned None or empty DataFrame for text [{i}]: '{single_text}'")
                 all_results.append({'original_input_text': single_text, 
                                     'sentence_analyzed': single_text, 
                                     'sentiment_label': 'N/A', 
                                     'sentiment_score': 0.0})
        logger.info(f"Sentiment analysis processing done. Generated {len(all_results)} sentence-level sentiment results.")
        return all_results
    except Exception as e:
        logger.error(f"Error during sentiment prediction loop: {e}", exc_info=True); return []

def run_news_sentiment_pipeline(user_stock_query, country='PK', language='en', desired_final_results=10): 
    logger.info(f"Starting news sentiment pipeline for user query: '{user_stock_query}', aiming for {desired_final_results} final results (filtering removed).")


    gnews_search_query = f'"{user_stock_query}" psx news'
    logger.info(f"Constructed GNews query: {gnews_search_query}")

    articles_to_fetch_from_gnews = desired_final_results + 5 
    news_articles_raw = scrape_news_for_stock(gnews_search_query, country, language, articles_to_fetch=articles_to_fetch_from_gnews)
    
    if not news_articles_raw:
        logger.warning("No news articles found from scraping.")
        return []

    articles_to_process_for_sentiment_payload = []
    for article_data in news_articles_raw:
        title = article_data.get('title', '').strip()
        description = article_data.get('description', '').strip()
        
        text_for_analysis = title
        if not title and description: text_for_analysis = description
        elif title and description and len(title) < 20 and len(description) > len(title): text_for_analysis = description

        if text_for_analysis: 
            articles_to_process_for_sentiment_payload.append({
                'text_to_analyze': text_for_analysis,
                'original_article': article_data
            })
        else:
            logger.warning(f"Article missing usable text, will be skipped for sentiment: {article_data.get('url')}")
        
        if len(articles_to_process_for_sentiment_payload) >= desired_final_results:
            break 
            
    logger.info(f"Selected {len(articles_to_process_for_sentiment_payload)} articles with usable text for sentiment analysis.")

    if not articles_to_process_for_sentiment_payload:
        logger.warning("No articles with usable text found for sentiment analysis.")
        return [] 

    texts_to_analyze_list = [item['text_to_analyze'] for item in articles_to_process_for_sentiment_payload]
    all_sentence_sentiments = analyze_sentiment_for_texts(texts_to_analyze_list, MODEL_PATH) 
    
    sentiments_by_original_text = {}
    for sent_res in all_sentence_sentiments:
        original_text = sent_res['original_input_text']
        if original_text not in sentiments_by_original_text: sentiments_by_original_text[original_text] = []
        sentiments_by_original_text[original_text].append(sent_res)

    final_pipeline_output = []
    for item_payload in articles_to_process_for_sentiment_payload: 
        original_article = item_payload['original_article']
        analyzed_text = item_payload['text_to_analyze']
        article_sentences_sentiments = sentiments_by_original_text.get(analyzed_text, [])
        
        final_sentiment_label = 'N/A'
        final_sentiment_score = 0.0

        if article_sentences_sentiments:
            total_score = sum(s['sentiment_score'] for s in article_sentences_sentiments)
            final_sentiment_score = total_score / len(article_sentences_sentiments)
            if final_sentiment_score > 0.15: final_sentiment_label = 'Positive'
            elif final_sentiment_score < -0.15: final_sentiment_label = 'Negative'
            else: final_sentiment_label = 'Neutral'
        
        combined_article = {
            'title': original_article.get('title', ''), 
            'description': original_article.get('description', ''),
            'published_date': original_article.get('published date', ''), 
            'url': original_article.get('url'),
            'publisher_name': original_article.get('publisher', {}).get('title', ''),
            'publisher_href': original_article.get('publisher', {}).get('href', ''),
            'analyzed_text': analyzed_text, 
            'sentiment_label': final_sentiment_label,
            'sentiment_score': final_sentiment_score, 
            'sentence_sentiments': article_sentences_sentiments 
        }
        final_pipeline_output.append(combined_article)
    
    logger.info(f"Pipeline processing complete. Returning {len(final_pipeline_output)} combined articles.")
    return final_pipeline_output


if __name__ == "__main__":
    if not PSX_SYMBOL_TO_NAME: 
        logger.error("Stock mappings (names.json) failed to load. This might affect symbol identification in the Django view for price fetching.")
    
    queries_to_test = ["LUCK", "K-Electric", "Pakistan stock market general news"]
    for user_query in queries_to_test:
        logger.info(f"\n\n<<<<<<<<<< RUNNING PIPELINE FOR USER QUERY: '{user_query}' >>>>>>>>>>")
        pipeline_results = run_news_sentiment_pipeline(user_query, desired_final_results=10) 
        if pipeline_results:
            logger.info(f"\n--- Pipeline Results for '{user_query}' (Count: {len(pipeline_results)}) ---")
            for item_index, item in enumerate(pipeline_results):
                print(f"\n--- Article {item_index + 1} ---")
                print(f"  Title: {item.get('title')}")
                print(f"  Published Date: {item.get('published_date')}")
                print(f"  Sentiment: {item.get('sentiment_label')} (Score: {item.get('sentiment_score', 0.0):.4f})")
            output_filename = f"{user_query.replace(' ', '_').replace('-', '_')}_sentiment_results_no_filter.json"
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(pipeline_results, f, indent=4, ensure_ascii=False)
            logger.info(f"\nFull results for '{user_query}' saved to {output_filename}")
        else:
            logger.info(f"No results obtained from the pipeline for '{user_query}'.")
        print("-" * 70)

