def uploadDataset(file_path=None):
    global dataset
    if file_path is None:
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        dataset = pd.read_csv(file_path)
        print("Dataset loaded successfully")
    else:
        print("No file selected")
import pandas as pd
from main import uploadDataset, dataset  # Adjust import if your main file has a different name

def test_uploadDataset():
    test_file_path = r"C:\Users\nagar\OneDrive\Desktop\Cyber Breaches copy\Dataset\Dataset.csv"
    uploadDataset(test_file_path)

    # Check if dataset loaded
    assert dataset is not None, "Dataset is None"
    assert not dataset.empty, "Dataset is empty"
    assert isinstance(dataset, pd.DataFrame), "Dataset is not a DataFrame"
    print("✅ test_uploadDataset passed.")
