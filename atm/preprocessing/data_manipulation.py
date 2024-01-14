from fire import Fire

def preprocess(data):
    data['col'] = 1
    return data

if __name__ == "__main__":
    Fire(preprocess)
