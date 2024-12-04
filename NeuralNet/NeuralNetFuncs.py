import numpy as np
import torch
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.keras.regularizers import l2

def season_to_numeric(season):
    seasons = {'Spring': 0, 'Summer': 1, 'Fall': 2, 'Winter': 3}
    return seasons.get(season, np.nan)

def preprocess_data(df):
    df_processed = df.copy()
    df_processed = df_processed.drop('id', axis=1)
    pciat_columns = [col for col in df_processed.columns if 'PCIAT' in col]
    df_processed = df_processed.drop(columns=pciat_columns)
    
    season_columns = [col for col in df_processed.columns if col.endswith('Season')]
    for col in season_columns:
        df_processed[col] = df_processed[col].apply(season_to_numeric)
    
    target = df_processed['sii']
    df_processed = df_processed.drop('sii', axis=1)
    
    mask = ~target.isna()
    df_processed = df_processed[mask]
    target = target[mask]
    
    numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
    df_processed[numeric_columns] = df_processed[numeric_columns].fillna(df_processed[numeric_columns].mean())
    
    return df_processed, target.astype(int)

def create_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim, kernel_regularizer=l2(0.01)),
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.2),
        Dense(4, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model




