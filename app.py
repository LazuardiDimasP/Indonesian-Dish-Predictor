import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('indonesian_dishes_expanded_with_images.csv')

# Convert the seasonings column from string to list
df['seasonings'] = df['seasonings'].apply(eval)

# Prepare the data
X = df[['protein', 'seasonings']]
y = df['dish']

# MultiLabelBinarizer for seasonings
mlb = MultiLabelBinarizer()
seasonings_binarized = mlb.fit_transform(X['seasonings'])

# Combine protein and seasonings for feature processing
X_combined = pd.concat([X['protein'], pd.DataFrame(seasonings_binarized, columns=mlb.classes_)], axis=1)

# OneHotEncoder for protein
preprocessor = ColumnTransformer(
    transformers=[
        ('protein', OneHotEncoder(), ['protein'])
    ], remainder='passthrough')

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Create a pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())  # You can change this to LogisticRegression() or GradientBoostingClassifier()
])

# Train the model
pipeline.fit(X_train, y_train)

# Streamlit app
st.title('Indonesian Dish Predictor')

# User inputs
protein = st.selectbox('Choose a protein', df['protein'].unique())
seasonings = st.multiselect('Choose seasonings', mlb.classes_)

# Predict
if st.button('Predict'):
    input_seasonings = [seasonings]
    input_seasonings_binarized = mlb.transform(input_seasonings)
    input_data = pd.DataFrame([[protein] + list(input_seasonings_binarized[0])], columns=['protein'] + list(mlb.classes_))
    prediction = pipeline.predict(input_data)
    predicted_dish = prediction[0]

    # Display predicted dish details
    dish_info = df[df['dish'] == predicted_dish].iloc[0]
    st.write(f'The predicted dish is: {predicted_dish}')
    st.write(f'Description: {dish_info["description"]}')
    st.image(dish_info["image_url"], width=300)

    # Display similar dishes
    similar_dishes = df[df['seasonings'].apply(lambda x: any(seasoning in x for seasoning in seasonings))]['dish'].unique()
    st.write('Similar dishes:', ', '.join(similar_dishes))