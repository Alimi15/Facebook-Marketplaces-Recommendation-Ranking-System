import pandas as pd
import clean_tabular_data
import clean_images
import model_train

if __name__ == "__main__":
    df = pd.read_csv("Products.csv", lineterminator="\n")
    df2 = pd.read_csv("Images.csv", lineterminator="\n")
    df = clean_tabular_data.clean(df)
    df, encoder = clean_tabular_data.encode(df)
    clean_images.clean_image_data("images/", df2)
    df_train = df.sample(frac=0.80)
    X_train = df_train[["product_name", "product_description", "location"]]
    y_train = df_train[["price"]]
    model_train.apply_tfidf(X_train, y_train)
    arr = clean_images.flatten_image()
    X = pd.DataFrame(arr)
    y = df["category"].values
    model_train.image_classification(X, y)
