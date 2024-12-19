import pandas as pd

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    test = False
    data = []
    for line in lines:
        parts = line.strip().split(' ::: ')
        if len(parts) == 4:
            test = False
            id_, title, genre, plot = parts
            data.append((id_, title, genre, plot))
        elif len(parts) == 3: 
            test = True
            id_, title, plot = parts
            data.append((id_, title, plot)) 
    if (test):
        return pd.DataFrame(data, columns=['id', 'title', 'plot'])
    else:
        return pd.DataFrame(data, columns=['id', 'title', 'genre', 'plot'])
