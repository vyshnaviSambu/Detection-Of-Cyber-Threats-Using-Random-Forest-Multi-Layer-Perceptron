from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import webbrowser
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import keras
from keras import layers
from keras.models import model_from_json
from keras.utils.np_utils import to_categorical
from keras.models import Model
from sklearn.neural_network import MLPClassifier
from keras.models import Model
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from tkinter import PhotoImage
from PIL import Image, ImageTk

global filename, autoencoder, decision_tree, dnn, encoder_model, pca
global X,Y
global dataset
global accuracy, precision, recall, fscore, vector
global X_train, X_test, y_train, y_test, scaler
labels = [
    'Phishing',
    'Denial of Service (DoS)',
    'Distributed Denial of Service (DDoS)',
    'Man-in-the-Middle (MITM)',
    'SQL Injection',
    'Cross-Site Scripting (XSS)',
    'Zero-Day Exploit',
    'Brute Force Attack'
]
main = tkinter.Tk()
main.title("CYBER THREAT DETECTION") #designing main screen
main.geometry("1300x1200")

#fucntion to upload dataset
def uploadDataset():
    global filename, dataset
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset") #upload dataset file
    text.insert(END,filename+" loaded\n\n")
    dataset = pd.read_csv(filename) #read dataset from uploaded file
    text.insert(END,"Dataset Values\n\n")
    text.insert(END,str(dataset.head()))
    text.update_idletasks()
    unique, count = np.unique(dataset['result'], return_counts=True)

    height = count
    bars = labels
    print(height)
    print(bars)
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.xticks(rotation=90)
    plt.title("Various Cyber-Attacks Found in Dataset") #plot graph with various attacks
    plt.show()
        
def preprocessing():
    text.delete('1.0', END)
    global dataset, scaler
    global X_train, X_test, y_train, y_test, X, Y
    
    dataset.fillna(0, inplace=True)  # Replace missing values with 0
    scaler = MinMaxScaler()  # Create new MinMaxScaler instance

    dataset = dataset.values
    X = dataset[:, 0:dataset.shape[1] - 1]
    Y = dataset[:, dataset.shape[1] - 1]

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)  # Shuffle dataset
    X = X[indices]
    Y = Y[indices]

    Y = to_categorical(Y)
    X = scaler.fit_transform(X)  # Fit & transform dataset with new scaler

    # Save the new scaler to avoid future issues
    with open('model/minmax.txt', 'wb') as file:
        pickle.dump(scaler, file)

    text.insert(END, "Dataset after feature normalization\n\n")
    text.insert(END, str(X) + "\n\n")
    text.insert(END, "Total records found in dataset: " + str(X.shape[0]) + "\n")
    text.insert(END, "Total features found in dataset: " + str(X.shape[1]) + "\n\n")

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END, "Dataset Train and Test Split\n\n")
    text.insert(END, "80% dataset records used to train ML algorithms: " + str(X_train.shape[0]) + "\n")
    text.insert(END, "20% dataset records used to test ML algorithms: " + str(X_test.shape[0]) + "\n")

def calculateMetrics(algorithm, predict, y_test):
    a = accuracy_score(y_test, predict) * 100
    p = precision_score(y_test, predict, average='macro', zero_division=1) * 100
    r = recall_score(y_test, predict, average='macro', zero_division=1) * 100
    f = f1_score(y_test, predict, average='macro', zero_division=1) * 100
    
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)

    text.insert(END, f"{algorithm} Accuracy  :  {a:.2f}%\n")
    text.insert(END, f"{algorithm} Precision : {p:.2f}%\n")
    text.insert(END, f"{algorithm} Recall    : {r:.2f}%\n")
    text.insert(END, f"{algorithm} FScore    : {f:.2f}%\n\n")

def runAutoEncoder():
    text.delete('1.0', END)
    global X_train, X_test, y_train, y_test, X, Y
    global autoencoder
    global accuracy, precision, recall, fscore
    accuracy = []
    precision = []
    recall = []
    fscore = []
    
    if os.path.exists("model/encoder_model.json"):
        with open('model/encoder_model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            autoencoder = model_from_json(loaded_model_json)
        json_file.close()
        autoencoder.load_weights("model/encoder_model_weights.h5")
        autoencoder._make_predict_function()
    else:
        encoding_dim = 32  # Further reduced encoding dimension
        input_size = keras.Input(shape=(X.shape[1],))
        
        # Adding Gaussian noise to input
        noisy_input = layers.GaussianNoise(0.3)(input_size)

        encoded = layers.Dense(encoding_dim, activation='linear')(noisy_input)  # Using linear activation
        encoded = layers.Dropout(0.7)(encoded)  # Increased dropout rate
        
        decoded = layers.Dense(y_train.shape[1], activation='softmax')(encoded)
        
        autoencoder = keras.Model(input_size, decoded)
        
        autoencoder.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        hist = autoencoder.fit(X_train, y_train, epochs=50, batch_size=128, shuffle=True, validation_data=(X_test, y_test))  # Reduced epochs and increased batch
        
        autoencoder.save_weights('model/encoder_model_weights.h5')          
        model_json = autoencoder.to_json()
        with open("model/encoder_model.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()

    print(autoencoder.summary())
    predict = autoencoder.predict(X_test)
    predict = np.argmax(predict, axis=1)
    testY = np.argmax(y_test, axis=1)
    
    calculateMetrics("AutoEncoder", predict, testY)

def runRandomForest():
    global autoencoder, random_forest, encoder_model, vector
    global X_train, X_test, y_train, y_test, X, Y, pca

    encoder_model = Model(autoencoder.inputs, autoencoder.layers[-1].output)  # creating autoencoder model
    vector = encoder_model.predict(X)  # extracting features using autoencoder
    pca = PCA(n_components=7)  # applying PCA for feature reduction
    vector = pca.fit_transform(vector)
    Y1 = np.argmax(Y, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(vector, Y1, test_size=0.2)
    random_forest = RandomForestClassifier()  # defining random forest
    random_forest.fit(X_train, y_train)  # training with random forest
    predict = random_forest.predict(X_test)
    text.insert(END, "Random Forest Trained\n")
    calculateMetrics("Random Forest", predict, y_test)

def runMLP():
    global autoencoder, mlp, encoder_model, vector
    global X_train, X_test, y_train, y_test, X, Y, pca

    encoder_model = Model(autoencoder.inputs, autoencoder.layers[-1].output)  # creating autoencoder model
    vector = encoder_model.predict(X)  # extracting features using autoencoder
    pca = PCA(n_components=7)  # applying PCA for feature reduction
    vector = pca.fit_transform(vector)
    Y1 = np.argmax(Y, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(vector, Y1, test_size=0.2)
    mlp = MLPClassifier()  # defining multilayer perceptron
    mlp.fit(X_train, y_train)  # training with MLP
    predict = mlp.predict(X_test)
    text.insert(END, "Multilayer Perceptron Trained \n")
    calculateMetrics("MLP", predict, y_test)

def attackAttributeDetection():
    text.delete('1.0', END)
    global autoencoder, decision_tree, encoder_model, pca, random_forest, mlp
    filename = filedialog.askopenfilename(initialdir="Dataset")
    dataset = pd.read_csv(filename)
    dataset.fillna(0, inplace=True)
    values = dataset.values
    temp = dataset.values
    temp = scaler.transform(temp)
    test_vector = encoder_model.predict(temp)  # extracting features using autoencoder
    test_vector = pca.transform(test_vector)
    print(test_vector.shape)
     
    # Predict using Random Forest
    predict_rf = random_forest.predict(test_vector)
    
    # Predict using MLP
    predict_mlp = mlp.predict(test_vector)

    for i in range(len(test_vector)):
       
        rf_prediction = predict_rf[i]
        mlp_prediction = predict_mlp[i]

       
        rf_result = "NO THREAT DETECTED" if rf_prediction == 0 else f"THREAT DETECTED Attribution Label: {labels[rf_prediction]}"
        mlp_result = "NO THREAT DETECTED" if mlp_prediction == 0 else f"THREAT DETECTED Attribution Label: {labels[mlp_prediction]}"

        result_text = (f"New Test Data : {str(values[i])}\n"
                       f"RF Result: {rf_result}\n"
                       f"MLP Result: {mlp_result}\n\n")
        
        text.insert(END, result_text)  

def graph(metric):
    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": fscore
    }
    
    df = pd.DataFrame({
        'Algorithms': ['AutoEncoder', 'Random Forest', 'MLP'],
        metric: metrics[metric]
    })
    
    df.plot(x='Algorithms', y=metric, kind='bar', legend=False)
    plt.title(f"{metric} Comparison")
    plt.ylabel(metric)
    plt.show()
    
def showGraphSelection():
    graph_window = Toplevel(main)
    graph_window.title("Select Metric to Plot")    
    Label(graph_window, text="Select Metric:", font=('times', 14, 'bold')).pack(pady=10)    
    Button(graph_window, text="Accuracy", command=lambda: graph("Accuracy"), font=('times', 12, 'bold')).pack(pady=5)
    Button(graph_window, text="Precision", command=lambda: graph("Precision"), font=('times', 12, 'bold')).pack(pady=5)
    Button(graph_window, text="Recall", command=lambda: graph("Recall"), font=('times', 12, 'bold')).pack(pady=5)
    Button(graph_window, text="F1 Score", command=lambda: graph("F1 Score"), font=('times', 12, 'bold')).pack(pady=5)

def comparisonTable():
    precautions = precautions = {
    "Phishing": "Implement email filtering systems, educate users to recognize phishing attempts, and enable multi-factor authentication (MFA) to prevent unauthorized access.",
    "Denial of Service (DoS)": "Implement rate-limiting, use firewalls and intrusion detection/prevention systems, and deploy load balancers to mitigate excessive traffic.",
    "Distributed Denial of Service (DDoS)": "Use DDoS protection services, deploy content delivery networks (CDNs), and have a well-prepared response plan for large-scale attacks.",
    "Man-in-the-Middle (MITM)": "Use strong encryption protocols (e.g., SSL/TLS), enforce secure connections, and implement certificate pinning to ensure authenticity of communication.",
    "SQL Injection": "Use parameterized queries, employ web application firewalls, sanitize and validate user inputs, and avoid dynamic SQL queries.",
    "Cross-Site Scripting (XSS)": "Sanitize user inputs, use content security policies (CSP), and implement secure coding practices such as output encoding.",
    "Zero-Day Exploit": "Keep software updated with security patches, use intrusion detection systems (IDS), and monitor for unusual network activity.",
    "Brute Force Attack": "Implement account lockout policies, use CAPTCHAs, enforce strong password policies, and utilize multi-factor authentication (MFA)."
}

    output = """
    <html>
    <head>
        <title>Cyber Threat Detection - Comparison Table</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background: linear-gradient(to right, #0066ff, #33ccff);
                text-align: center;
                color: white;
            }}
            h2 {{
                color: #ffcc00;
                text-shadow: 2px 2px 4px #000000;
            }}
            table {{
                width: 80%;
                margin: auto;
                border-collapse: collapse;
                background-color: rgba(255, 255, 255, 0.9);
                color: black;
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.5);
            }}
            th, td {{
                border: 1px solid black;
                padding: 10px;
                text-align: center;
                font-weight: bold;
            }}
            th {{
                background-color: #ff6600;
                color: white;
            }}
            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            tr:hover {{
                background-color: #ffff99;
            }}
            h3 {{
                margin-top: 40px;
                color: #ffcc00;
                text-shadow: 2px 2px 4px #000000;
            }}
            .precaution-list {{
                width: 80%;
                margin: auto;
                text-align: left;
                background-color: rgba(255, 255, 255, 0.9);
                color: black;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.5);
            }}
            li {{
                margin: 10px 0;
                font-size: 16px;
                font-weight: bold;
                color: #333;
            }}
        </style>
    </head>
    <body>
        <h2>Cyber Threat Detection - Performance Comparison</h2>
        <table>
            <tr>
                <th>Algorithm Name</th>
                <th>Accuracy (%)</th>
                <th>Precision (%)</th>
                <th>Recall (%)</th>
                <th>F1 Score (%)</th>
            </tr>
            <tr>
                <td>AutoEncoder</td>
                <td>{0:.2f}</td>
                <td>{1:.2f}</td>
                <td>{2:.2f}</td>
                <td>{3:.2f}</td>
            </tr>
            <tr>
                <td>Random Forest</td>
                <td>{4:.2f}</td>
                <td>{5:.2f}</td>
                <td>{6:.2f}</td>
                <td>{7:.2f}</td>
            </tr>
            <tr>
                <td>MLP</td>
                <td>{8:.2f}</td>
                <td>{9:.2f}</td>
                <td>{10:.2f}</td>
                <td>{11:.2f}</td>
            </tr>
        </table>

        <h3>Precautionary Measures for Cyber Threats</h3>
        <div class="precaution-list">
            <ul>
    """.format(accuracy[0], precision[0], recall[0], fscore[0],
               accuracy[1], precision[1], recall[1], fscore[1],
               accuracy[2], precision[2], recall[2], fscore[2])

    for attack, precaution in precautions.items():
        output += f"<li><b>{attack}:</b> {precaution}</li>"

    output += """
            </ul>
        </div>
    </body>
    </html>
    """
    with open("table.html", "w") as f:
        f.write(output)

    webbrowser.open("table.html", new=2)

bg_img = Image.open("image.jpg")  
bg_img = bg_img.resize((1300, 1200), Image.ANTIALIAS)  

bg_img_tk = ImageTk.PhotoImage(bg_img)

bg_label = Label(main, image=bg_img_tk)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

font = ('times', 14, 'bold')
title = Label(main, text='CyberGuard:predicting and preventing cyber breaches using Autoencoder, Random forest, Multi layer perceptron', bg='LightSkyBlue', fg='black')
title.config(bg='greenyellow', fg='dodger blue')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Dataset", command=uploadDataset)
uploadButton.place(x=50,y=550)
uploadButton.config(font=font1)  

processButton = Button(main, text="Preprocess Dataset", command=preprocessing)
processButton.place(x=330,y=550)
processButton.config(font=font1) 

autoButton = Button(main, text="Run AutoEncoder Algorithm", command=runAutoEncoder)
autoButton.place(x=730,y=550)
autoButton.config(font=font1)

dtButton = Button(main, text="Run Random Forest", command=runRandomForest)
dtButton.place(x=1030,y=550)
dtButton.config(font=font1)

dnnButton = Button(main, text="Run MLP", command=runMLP)
dnnButton.place(x=50,y=600)
dnnButton.config(font=font1)

attributeButton = Button(main, text="Detection & Attribute Attack Type", command=attackAttributeDetection)
attributeButton.place(x=330,y=600)
attributeButton.config(font=font1) 

graphButton = Button(main, text="Comparison Graph", command=showGraphSelection)
graphButton.place(x=730,y=600)
graphButton.config(font=font1)

tableButton = Button(main, text="Comparison Table", command=comparisonTable)
tableButton.place(x=1030,y=600)
tableButton.config(font=font1)

#main.config(bg='LightSkyBlue')
main.mainloop()
