import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix

if __name__ == "__main__":
    print("Running neuralNetwork.py...")

    # PART 1 - READING AND PREPROCESSING DATA

    # Define the path to your PCAP spreadsheet file
    file_path = 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'

    # Read data from the PCAP spreadsheet file
    try:
        data = pd.read_csv(file_path)
        print("Data successfully loaded.")

        # Step 2: Parse the specified columns for features and labels
        selected_columns = [
            " Flow Duration",
            " Bwd Packet Length Std",
            "Active Mean",
            " Flow IAT Std",
            " Subflow Fwd Bytes",
            "Total Length of Fwd Packets",
            " Label"
        ]
        parsed_data = data[selected_columns]

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        exit()  # Exit if the file is not found

    # STEP 2 - SPLIT THE DATA INTO TRAINING AND TESTING SETS

    # Define the feature columns and target column
    feature_columns = [
        " Flow Duration",
        " Bwd Packet Length Std",
        "Active Mean",
        " Flow IAT Std",
        " Subflow Fwd Bytes",
        "Total Length of Fwd Packets"
    ]
    target_column = " Label"

    # Split the data into features (X) and labels (y)
    X = parsed_data[feature_columns]
    y = parsed_data[target_column]

    # Convert string labels to numeric values
    y = y.map({'BENIGN': 0, 'DDoS': 1})

    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 1 - Normalize the feature data
    scaler = StandardScaler()  # Create a StandardScaler instance
    X_train = scaler.fit_transform(X_train)  # Fit and transform the training data
    X_test = scaler.transform(X_test)  # Transform the test data

    # Display the shapes of the training and testing sets
    print("Training data shape:", X_train.shape)
    print("Testing data shape:", X_test.shape)

    # STEP 3 - MODEL DEVELOPMENT, TRAINING, AND EVALUATION

    # Define a simple neural network model with slight modifications
    model = keras.Sequential([
        keras.layers.Dense(units=64, activation='relu', input_shape=(X_train.shape[1],)),  # Changed input_dim to input_shape
        keras.layers.Dropout(0.5),  # Dropout layer added for regularization
        keras.layers.Dense(units=32, activation='relu'),
        keras.layers.Dropout(0.5),  # Another Dropout layer
        keras.layers.Dense(units=1, activation='sigmoid')  # Output layer for binary classification
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model with an increased number of epochs
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)  # Increased epochs from 10 to 20

    # STEP 4 - EVALUATION OF MODEL PERFORMANCE

    # Generate prediction model
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)  # Convert predictions to binary

    # Print evaluation metrics
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_binary))

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_binary))
