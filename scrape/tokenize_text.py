from transformers import CLIPTokenizer

# Load the CLIP tokenizer
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# Define the strings you want to tokenize
strings = [
    "Let's see how CLIP tokenizes this text!",
    "source_furry, anthro, coyote, HX, black and green topwear, gay furry boi"
]

# Tokenize the strings
for string in strings:
    # Tokenize the string
    tokens = tokenizer.tokenize(string)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    # Print the original string
    print("Original string:", string)

    # Print the tokenized string
    print("Tokenized string:", tokens)

    # Print the token IDs
    print("Token IDs:", token_ids)

    # Print the decoded string
    decoded_string = tokenizer.decode(token_ids)
    print("Decoded string:", decoded_string)

    print("---")
