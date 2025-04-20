import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string
from collections import defaultdict

def process_restaurant_reviews(input_csv_path, output_csv_path, restaurant_col='name', review_col='text', 
                              categories_col='categories', review_id_col='review_id', business_id_col='business_id'):

    df = pd.read_csv(input_csv_path)
    restaurant_mask = df[categories_col].str.lower().str.contains('restaurant', na=False)
    restaurant_df = df[restaurant_mask]

    
    if len(restaurant_df) == 0:
        return None

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    def preprocess_text(text):
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_clean(text):
        
        cleaned_text = preprocess_text(text)
        tokens = word_tokenize(cleaned_text)
        clean_tokens = [lemmatizer.lemmatize(word) for word in tokens 
                        if word not in stop_words and len(word) > 1]
        
        return clean_tokens
    
    restaurant_df['tokenized_review'] = restaurant_df[review_col].apply(tokenize_and_clean)
    
    restaurant_word_data = {}
    restaurant_business_ids = {}
    
    for restaurant, restaurant_group in restaurant_df.groupby(restaurant_col):
        word_data = defaultdict(lambda: [0])

        business_id = restaurant_group[business_id_col].iloc[0]
        restaurant_business_ids[restaurant] = business_id
        
        for _, row in restaurant_group.iterrows():
            review_id = row[review_id_col]
            tokens = row['tokenized_review']
            
            for token in tokens:
                if review_id not in word_data[token]:
                    word_data[token][0] += 1
                    word_data[token].append(review_id)
        
        restaurant_word_data[restaurant] = dict(word_data)
    
    result_data = []
    for restaurant, word_data in restaurant_word_data.items():
        result_data.append({
            restaurant_col: restaurant,
            business_id_col: restaurant_business_ids[restaurant],
            'word_data': word_data,
            'unique_word_count': len(word_data),
        })
    
    output_df = pd.DataFrame(result_data)
    
    print(f"Saving data to {output_csv_path}...")
    output_df.to_csv(output_csv_path, index=False)
    return output_df

# Example usage
if __name__ == "__main__":
    input_path = "Nashville.csv"
    output_path = "Nashville_review_tokens2.csv"
 
    results = process_restaurant_reviews(
        input_path, 
        output_path, 
        restaurant_col="name",       
        review_col="text",          
        categories_col="categories", 
        review_id_col="review_id"  
    )

    input_path = "NewOrleans.csv"
    output_path = "NewOrleans_review_tokens2.csv"
    
    results = process_restaurant_reviews(
        input_path, 
        output_path, 
        restaurant_col="name",   
        review_col="text",          
        categories_col="categories", 
        review_id_col="review_id"  
    )

    input_path = "Philadelphia.csv"
    output_path = "Philadelphia_review_tokens2.csv"

    results = process_restaurant_reviews(
        input_path, 
        output_path, 
        restaurant_col="name",    
        review_col="text",           
        categories_col="categories", 
        review_id_col="review_id"  
    )

    input_path = "Tampa.csv"
    output_path = "Tampa_review_tokens2.csv"
    
    results = process_restaurant_reviews(
        input_path, 
        output_path, 
        restaurant_col="name",      
        review_col="text",         
        categories_col="categories",
        review_id_col="review_id"  
    )

    input_path = "Tucson.csv"
    output_path = "Tucson_review_tokens2.csv"
    

    results = process_restaurant_reviews(
        input_path, 
        output_path, 
        restaurant_col="name",      
        review_col="text",           
        categories_col="categories",
        review_id_col="review_id" 
    )

    input_path = "Indianapolis.csv"
    output_path = "Indianapolis_review_tokens2.csv"
    
    results = process_restaurant_reviews(
        input_path, 
        output_path, 
        review_col="text",         
        categories_col="categories", 
        review_id_col="review_id"   
    )

    