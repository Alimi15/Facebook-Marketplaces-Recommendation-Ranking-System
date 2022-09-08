import pandas as pd
import clean_tabular_data
import clean_images

if __name__ == "__main__":
    df = pd.read_csv("Products.csv", lineterminator="\n")
    df = clean_tabular_data.remove_missing_rows(df)
    clean_images.clean_image_data("images/")
