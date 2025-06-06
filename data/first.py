import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

STOP_WORDS = ['и', 'в', 'на', 'с', 'для', 'к', 'по', 'или', 'a']
TEXT_COLUMNS = ['Название', 'Детальное описание', 'Преимущества', 'Материал', 'Бренд']
CATEGORY_COLUMNS = ['cat_0', 'cat_1', 'cat_3']
MODEL_PARAMS = {'n_estimators': 100, 'random_state': 42}

def load_data():
    return (
        pd.read_excel("Данные поставщика.xlsx"),
        pd.read_excel("Дерево категорий.xlsx"),
        pd.read_excel("Список товаров.xlsx")
    )

def create_combined_text(df, columns, new_col):
    df[new_col] = df[columns].fillna('').agg(' '.join, axis=1)
    return df

def prepare_text_data(supplier, categories):
    supplier = create_combined_text(supplier, TEXT_COLUMNS, 'text')
    categories = create_combined_text(categories, CATEGORY_COLUMNS, 'category_text')
    return supplier, categories

def vectorize_texts(texts, supplier_len):
    vectorizer = TfidfVectorizer(stop_words=STOP_WORDS)
    vectors = vectorizer.fit_transform(texts)
    return vectors[:supplier_len], vectorizer

def train_and_predict(vectors, supplier_ids, items, clf=None):
    merged = pd.DataFrame({'item_id': supplier_ids}).merge(items, on='item_id', how='left')
    train_mask = merged['cat_id'].notna()
    
    if train_mask.sum() < 2:
        return pd.DataFrame({'item_id': supplier_ids, 'predicted_cat_id': 0}), None
    
    if clf is None:
        clf = RandomForestClassifier(**MODEL_PARAMS)
        clf.fit(vectors[train_mask], merged.loc[train_mask, 'cat_id'])
    
    return pd.DataFrame({
        'item_id': supplier_ids,
        'predicted_cat_id': clf.predict(vectors)
    }), clf

def evaluate_predictions(result, items):
    merged = result.merge(items, on='item_id', how='left')
    correct = (merged['predicted_cat_id'] == merged['cat_id']).sum()
    total = len(merged)
    print(f"Correct: {correct}, Incorrect: {total - correct}")
    print(f"Accuracy: {correct / total:.2f}")

def predict_single_category(item_id, supplier, vectorizer, clf):
    if item_id not in supplier['Код артикула'].values:
        return f"Error: Item ID {item_id} not found"
    
    item_index = supplier[supplier['Код артикула'] == item_id].index[0]
    item_text = supplier.loc[item_index, 'text']
    item_vector = vectorizer.transform([item_text])
    predicted_category = clf.predict(item_vector)[0]
    
    return predicted_category

def main():
    supplier, categories, items = load_data()
    supplier, categories = prepare_text_data(supplier, categories)
    all_texts = supplier['text'].tolist() + categories['category_text'].tolist()
    vectors, vectorizer = vectorize_texts(all_texts, len(supplier))
    result, clf = train_and_predict(vectors, supplier['Код артикула'], items)
    evaluate_predictions(result, items)
    result.to_excel("predicted_categories.xlsx", index=False)
    
    return supplier, vectorizer, clf

if __name__ == "__main__":
    supplier, vectorizer, clf = main()
    item_id = int(input("Введите код товара для предсказания категории: "))
    predicted_category = predict_single_category(item_id, supplier, vectorizer, clf)
    print(f"Предсказанная категория товара {item_id}: {predicted_category}")