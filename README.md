# machine_learning
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn

# Presupunem că datele sunt încărcate într-un DataFrame pandas
# date = pd.read_csv('data.csv')
# Exemplu de date
data = {
    'description': ['Great game with amazing graphics', 'Exciting gameplay and story', 'Average puzzle game', 'Top-notch RPG with great mechanics'],
    'year': [2020, 2019, 2018, 2021],
    'price': [59.99, 49.99, 29.99, 69.99]
}
df = pd.DataFrame(data)

# Preprocesarea textului
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_descriptions(descriptions):
    return tokenizer(descriptions, padding=True, truncation=True, return_tensors="pt")

tokenized_descriptions = tokenize_descriptions(df['description'].tolist())

# Preprocesarea anului
scaler = StandardScaler()
years_scaled = scaler.fit_transform(df[['year']].values)

# Construirea modelului
class PricePredictor(nn.Module):
    def _init_(self):
        super(PricePredictor, self)._init_()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.regressor = nn.Sequential(
            nn.Linear(768 + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, input_ids, attention_mask, years):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = bert_outputs.last_hidden_state[:, 0, :]
        combined_input = torch.cat((cls_output, years), dim=1)
        price = self.regressor(combined_input)
        return price

model = PricePredictor()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Împărțirea datelor
X_train, X_test, y_train, y_test = train_test_split(
    tokenized_descriptions['input_ids'], df['price'].values, test_size=0.2, random_state=42
)
X_train_mask, X_test_mask = train_test_split(tokenized_descriptions['attention_mask'], test_size=0.2, random_state=42)
X_train_years, X_test_years = train_test_split(years_scaled, test_size=0.2, random_state=42)

# Transformarea datelor în tensori
X_train, X_train_mask, X_train_years = map(torch.tensor, (X_train.numpy(), X_train_mask.numpy(), X_train_years))
X_test, X_test_mask, X_test_years = map(torch.tensor, (X_test.numpy(), X_test_mask.numpy(), X_test_years))
y_train = torch.tensor(y_train).float().view(-1, 1)
y_test = torch.tensor(y_test).float().view(-1, 1)

# Antrenarea modelului
max_epochs = 1000
patience = 10
best_mae = float('inf')
patience_counter = 0

for epoch in range(max_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train, X_train_mask, X_train_years)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    # Evaluarea pe setul de testare
    model.eval()
    with torch.no_grad():
        predictions = model(X_test, X_test_mask, X_test_years)
        mae = mean_absolute_error(y_test.numpy(), predictions.numpy())
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}, MAE: {mae}')
    
    # Condiție de oprire
    if mae < best_mae:
        best_mae = mae
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print("Early stopping")
        break

print(f'Best MAE: {best_mae}')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn

# Presupunem că datele sunt încărcate într-un DataFrame pandas
# date = pd.read_csv('data.csv')
# Exemplu de date
data = {
    'description': ['Great game with amazing graphics', 'Exciting gameplay and story', 'Average puzzle game', 'Top-notch RPG with great mechanics'],
    'year': [2020, 2019, 2018, 2021],
    'price': [59.99, 49.99, 29.99, 69.99]
}
df = pd.DataFrame(data)

# Preprocesarea textului
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_descriptions(descriptions):
    return tokenizer(descriptions, padding=True, truncation=True, return_tensors="pt")

tokenized_descriptions = tokenize_descriptions(df['description'].tolist())

# Preprocesarea anului
scaler = StandardScaler()
years_scaled = scaler.fit_transform(df[['year']].values)

# Construirea modelului
class PricePredictor(nn.Module):
    def _init_(self):
        super(PricePredictor, self)._init_()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.regressor = nn.Sequential(
            nn.Linear(768 + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, input_ids, attention_mask, years):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = bert_outputs.last_hidden_state[:, 0, :]
        combined_input = torch.cat((cls_output, years), dim=1)
        price = self.regressor(combined_input)
        return price

model = PricePredictor()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Împărțirea datelor
X_train, X_test, y_train, y_test = train_test_split(
    tokenized_descriptions['input_ids'], df['price'].values, test_size=0.2, random_state=42
)
X_train_mask, X_test_mask = train_test_split(tokenized_descriptions['attention_mask'], test_size=0.2, random_state=42)
X_train_years, X_test_years = train_test_split(years_scaled, test_size=0.2, random_state=42)

# Transformarea datelor în tensori
X_train, X_train_mask, X_train_years = map(torch.tensor, (X_train, X_train_mask, X_train_years))
X_test, X_test_mask, X_test_years = map(torch.tensor, (X_test, X_test_mask, X_test_years))
y_train = torch.tensor(y_train).float().view(-1, 1)
y_test = torch.tensor(y_test).float().view(-1, 1)

# Antrenarea modelului
epochs = 10
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train, X_train_mask, X_train_years)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# Evaluarea modelului
model.eval()
with torch.no_grad():
    predictions = model(X_test, X_test_mask, X_test_years)
    mae = mean_absolute_error(y_test.numpy(), predictions.numpy())
    print(f'Mean Absolute Error: {mae}')

    import pandas as pd
import matplotlib.pyplot as plt

# Încărcarea setului de date
df = pd.read_csv('https://storage.googleapis.com/datasets-3456/loan_train_data.csv')

# Vizualizarea datelor
print(df.head())

# Vizualizarea coloanei 'ApplicantIncome'
print(df['ApplicantIncome'])

# Multiplicarea valorilor din coloana 'ApplicantIncome' cu 2 (doar pentru vizualizare)
print(df['ApplicantIncome'] * 2)

# Filtrarea aplicațiilor cu venituri mai mari de 5000
print(df['ApplicantIncome'] > 5000)
print(df['ApplicantIncome'].loc[df['ApplicantIncome'] > 5000])

# Vizualizarea coloanei 'Property_Area'
print(df['Property_Area'])

# Afișarea valorilor unice și a distribuției acestora în 'Property_Area'
print(df['Property_Area'].unique())
print(df['Property_Area'].value_counts())

# Crearea unui barplot pentru distribuția 'Property_Area'
plt.bar([0, 1, 2], df['Property_Area'].value_counts())
plt.show()

# Crearea unui histogram pentru 'ApplicantIncome'
plt.hist(df['ApplicantIncome'], bins=80)
plt.show()

# Vizualizarea valorilor unice din coloana 'Gender'
print(df['Gender'].unique())  # NaN -> Not a Number

# Adăugarea unei noi coloane 'GE' și setarea inițială la 0
df['GE'] = 0
print(df.head())

# Setarea valorilor în 'GE' pe baza genului
df.loc[df['Gender'] == 'Female', 'GE'] = 1
df.loc[df['Gender'] == 'Male', 'GE'] = 2
print(df.head())

# Eliminarea coloanelor 'Loan_ID' și 'Gender'
df.drop(columns=['Loan_ID', 'Gender'], inplace=True)
print(df.head())

# Vizualizarea distribuției 'Married' și maparea la valori numerice
print(df['Married'].value_counts())
df['Married'] = df['Married'].map({'No': 0, 'Yes': 1})
print(df.head())

# Vizualizarea distribuției 'Dependents' și maparea la valori numerice
print(df['Dependents'].value_counts())
df['Dependents'] = df['Dependents'].map({'0': 0, '1': 1, '2': 2, '3+': 3})
print(df.head())

# Vizualizarea distribuției 'Education' și maparea la valori numerice
print(df['Education'].value_counts())
df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
print(df.head())

# Vizualizarea distribuției 'Property_Area' și aplicarea one-hot encoding
print(df['Property_Area'].value_counts())
df = pd.get_dummies(df, columns=['Property_Area'])
print(df.info())

# Vizualizarea coloanei 'LoanAmount' și tratarea valorilor lipsă
print(df['LoanAmount'])
print(df['LoanAmount'].mean())
print(df['LoanAmount'].isna())
df.loc[df['LoanAmount'].isna(), 'LoanAmount'] = df['LoanAmount'].mean()
print(df.info())

# Tratarea valorilor lipsă pentru 'Loan_Amount_Term' și 'Credit_History'
df.loc[df['Loan_Amount_Term'].isna(), 'Loan_Amount_Term'] = df['Loan_Amount_Term'].mean()
df.loc[df['Credit_History'].isna(), 'Credit_History'] = df['Credit_History'].mean()
print(df.head())

# Definirea matricei de caracteristici (X) și a vectorului de răspuns (y)
X = df[[
    'Married',
    'Dependents',
    'Education',
    'ApplicantIncome',
    'CoapplicantIncome',
    'LoanAmount',
    'Loan_Amount_Term',
    'Credit_History',
    'Property_Area_Rural',
    'Property_Area_Semiurban',
    'Property_Area_Urban',
]]
print(X)
print(X.info())

# Tratarea valorilor lipsă în 'Dependents' și 'Married'
X.loc[X['Dependents'].isna(), 'Dependents'] = X['Dependents'].mode()[0]
X.loc[X['Married'].isna(), 'Married'] = X['Married'].mode()[0]
print(X.info())

# Definirea vectorului de răspuns (y) prin maparea valorilor 'Loan_Status'
y = df['Loan_Status'].map({'Y': 1, 'N': 0})
print(y)

# Importarea și antrenarea unui model KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(X, y)
print("KNeighborsClassifier accuracy:", model.score(X, y))

# Importarea și antrenarea unui model RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X, y)
print("RandomForestClassifier accuracy:", model.score(X, y))

# Împărțirea setului de date în train și test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

# Antrenarea și evaluarea modelului RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
print("Training accuracy:", model.score(X_train, y_train))
print("Testing accuracy:", model.score(X_test, y_test))

plt.figure(figsize=(10, 5))
sns.boxplot(x=df['price'])
plt.show()
df['price'] = np.where((df['price'] < lower_bound) | (df['price'] > upper_bound), df['price'].median(), df['price'])
print("Data with outliers replaced by median:")
print(df)
#outliers

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

# Setările directoarelor pentru datele de antrenament și validare
train_dir = 'path/to/train_data'
validation_dir = 'path/to/validation_data'

# Setarea generatorului de date pentru augmentarea imaginilor
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Generator de date pentru setul de antrenament
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Generator de date pentru setul de validare
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Utilizarea unui model preantrenat ResNet50 pentru extragerea caracteristicilor
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Adăugarea unor straturi suplimentare pentru clasificare
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

# Crearea modelului final
model = Model(inputs=base_model.input, outputs=predictions)

# Congelarea straturilor de bază (pentru a păstra greutățile preantrenate)
for layer in base_model.layers:
    layer.trainable = False

# Compilarea modelului
model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Antrenarea modelului
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=10
)

# Evaluarea modelului
validation_loss, validation_accuracy = model.evaluate(validation_generator, steps=validation_generator.samples // validation_generator.batch_size)
print(f'Validation accuracy: {validation_accuracy}')

# Plotarea istoricului de antrenament
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
