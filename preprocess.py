#!/usr/bin/python3
from pipeline import preprocess, generate

if __name__ == "__main__":
    data = preprocess()
    data.to_csv('data/phase_12_filled.csv', index=False)
    data = generate()
