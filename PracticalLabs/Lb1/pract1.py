import nltk
from nltk.tokenize import word_tokenize

# Ensure punkt is downloaded
nltk.download('punkt')
nltk.download('punkt_tab')

def tokenize_text_file(file_path):
    """Tokenizes the text in the given file."""
    try:
        with open(file_path, 'r') as file:
            text = file.read()

        # Tokenize the text into words
        tokens = word_tokenize(text)
        return tokens

    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return []

if __name__ == "__main__":
    # Specify the path to your text file
    file_path = r"F:\M.Tech_CollgeMaterials\CodeLLM\PracticalLabs\Lb1\Sample.txt"  # Update the path

    # Call the function to tokenize the file content
    tokens = tokenize_text_file(file_path)

    if tokens:
        print(tokens)
