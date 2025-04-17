import os
import base64
from dotenv import load_dotenv
from together import Together
from tkinter import Tk, filedialog

# Load environment variables from the .env file
load_dotenv(r"C:\Users\Fergus Watson\OneDrive\Desktop\Python Web\LLama OCR\3.2-90B-Vision\another_api_key.env")

# Get the API key from .env file
my_api_key = os.getenv("TOGETHER_API_KEY")

# Checks if my_api_exists (is not an empty string)
# Raised ValueError in order to hault program if condition met and prints api key found if good
if not my_api_key:
    raise ValueError("API Key not found. Please check your .env file.")
else:
    print("API key found")
    
# Checks that api_key = my_api_key whcih enables communication with together API 
# Creating client as an object and then enables access to together class thus communication
client = Together(api_key=my_api_key)


# API requires image to be sent as Base64 encoded string
# Function to convert the image into binary data (raw bytes) then base 64 string 
def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# Defines select_image function to take image from computer files and uses filedialog.askopenfilename to open documents
# filetypes filters file types so only selected types are visible for selection
def select_image():
    image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.webp")])
    return image_path

# Calls select image function and assigns it to image_path
image_path = select_image()

# Conditional check to verify image_path value - essentially checks that image_path has a value
if not image_path:
    print("No image selected. Exiting.")
    exit()

# Prompt given to model
getDescriptionPrompt = "What is the meter reading and meter serial number on this meter?"  \
"Submit the response in the following format: Meter Reading: | Meter Serial Number: "    

# used try to deal with potential errors in model running
# Sends the base 64 image request to the AI API. 
# Defines model as 3.2 11B Turbo
# messages part deals with what is being sent to the AI
# First it specifies that the message is coming from the user ("role":"user")
# Splits the message into two elements: 1) the text input 2) base64 image
# stream=True instantly submits responses in chunks thus faster.
# if data stored from each model probably want to stream=false 
try:
    base64_image = encode_image(image_path)

    stream = client.chat.completions.create(
        model="meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": getDescriptionPrompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            }
        ],
        stream=True,
        temperature = 0,
    )
    
    #\n just returns the atual extrcated text on the line bellow
    print("Extracted Text:\n ")

    # Improved stream handling with proper checks
    for chunk in stream:
        # Check if chunk has choices
        if not hasattr(chunk, 'choices') or not chunk.choices:
            continue

        # Check if first choice exists and has delta
        choice = chunk.choices[0]
        
        if not hasattr(choice, 'delta'):
            continue

        # Check if delta has content
        delta = choice.delta
        if not hasattr(delta, 'content'):
            continue

        # Print content if it exists
        content = delta.content
        if content is not None:
            print(content, end="", flush=True)

# prints fileNotFoundError if error
except FileNotFoundError:
    print(f"Error: Could not find image file at {image_path}")

# except Exception as e prints specific error message
except Exception as e:
    print(f"An error occurred: {str(e)}")
