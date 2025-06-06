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
    return supplier['text'].tolist() + categories['category_text'].tolist()


def vectorize_texts(texts, supplier_len):
    vectorizer = TfidfVectorizer(stop_words=STOP_WORDS)
    return vectorizer.fit_transform(texts)[:supplier_len]


def train_and_predict(vectors, supplier_ids, items):
    merged = pd.DataFrame({'item_id': supplier_ids}).merge(items, on='item_id', how='left')
    train_mask = merged['cat_id'].notna()
    
    if train_mask.sum() < 2:
        return pd.DataFrame({'item_id': supplier_ids, 'predicted_cat_id': 0})
    
    clf = RandomForestClassifier(**MODEL_PARAMS)
    clf.fit(vectors[train_mask], merged.loc[train_mask, 'cat_id'])
    return pd.DataFrame({
        'item_id': supplier_ids,
        'predicted_cat_id': clf.predict(vectors)
    })


def evaluate_predictions(result, items):
    merged = result.merge(items, on='item_id', how='left')
    correct = (merged['predicted_cat_id'] == merged['cat_id']).sum()
    total = len(merged)
    print(f"Correct: {correct}, Incorrect: {total - correct}")
    print(f"Accuracy: {correct / total:.2f}")


def main():
    supplier, categories, items = load_data()
    all_texts = prepare_text_data(supplier, categories)
    vectors = vectorize_texts(all_texts, len(supplier))
    result = train_and_predict(vectors, supplier['Код артикула'], items)
    evaluate_predictions(result, items)
    result.to_excel("predicted_categories.xlsx", index=False)


if __name__ == "__main__":
    main()