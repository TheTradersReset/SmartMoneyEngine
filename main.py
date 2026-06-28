from src.core.data_loader import DataLoader

loader = DataLoader()

df = loader.load_csv("data/sample/nifty_sample.csv")

print(df)