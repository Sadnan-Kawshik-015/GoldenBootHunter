import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import pandas as pd
from io import BytesIO
from PIL import Image

MODEL = tf.keras.models.load_model("Model/FootBallerMobileNet.h5")

# st.set_page_config(
#     page_title="SportLens: Analyzing Sports Images",
#     page_icon=":soccer:"
# )
# CLASS_NAME = ['Air Hockey', 'Ampute Football', 'Archery', 'Arm Wrestling', 'Axe Throwing', 'Balance Beam', 'Barell Racing', 'Baseball', 'Basketball', 'Baton Twirling', 'Bike Polo', 'Billiards', 'BMX', 'Bobsled', 'Bowling', 'Boxing', 'Bull Riding', 'Bungee Jumping', 'Canoe Slamon', 'Cheerleading', 'Chuckwagon Racing', 'Cricket', 'Croquet', 'Curling', 'Disc Golf', 'Fencing', 'Field Hockey', 'Figure Skating Men', 'Figure Skating Pairs', 'Figure Skating Women', 'Fly Fishing', 'Football', 'Formula 1 Racing', 'Frisbee', 'Gaga', 'Giant Slalom', 'Golf', 'Hammer Throw', 'Hang Gliding', 'Harness Racing', 'High Jump', 'Hockey', 'Horse Jumping', 'Horse Racing', 'Horseshoe Pitching', 'Hurdles', 'Hydroplane Racing', 'Ice Climbing', 'Ice Yachting', 'Jai Alai', 'Javelin', 'Jousting', 'Judo', 'Lacrosse', 'Log Rolling', 'Luge', 'Motorcycle Racing', 'Mushing', 'NASCAR Racing', 'Olympic Wrestling', 'Parallel Bar', 'Pole Climbing', 'Pole Dancing', 'Pole Vault', 'Polo', 'Pommel Horse', 'Rings', 'Rock Climbing', 'Roller Derby', 'Rollerblade Racing', 'Rowing', 'Rugby', 'Sailboat Racing', 'Shot Put', 'Shuffleboard', 'Sidecar Racing', 'Ski Jumping', 'Sky Surfing', 'Skydiving', 'Snow Boarding', 'Snowmobile Racing', 'Speed Skating', 'Steer Wrestling', 'Sumo Wrestling', 'Surfing', 'Swimming', 'Table Tennis', 'Tennis', 'Track Bicycle', 'Trapeze', 'Tug Of War', 'Ultimate', 'Uneven Bars', 'Volleyball', 'Water Cycling', 'Water Polo', 'Weightlifting', 'Wheelchair Basketball', 'Wheelchair Racing', 'Wingsuit Flying']

CLASS_NAMES = ['alessandro_del_piero',
              'andreas_iniesta',
              'andriy_shevchenko',
              'cristiano_ronaldo',
              'didier_drogba',
              'diego_maradona',
              'edinson_cavani',
              'francesco_totti',
              'gianlugi_buffon',
              'iker_casillas',
              'lionel_messi',
              'luka_modric',
              'mohamed_salah',
              'pavel_nedved',
              'pele',
              'riyan_giggs',
              'roberto_baggio',
              'roberto_carlos',
              'ronaldinho',
              'ronaldo_nazario',
              'samuel_eto',
              'zlatan_ibrahimovic']


def read_file_as_image(data):
    image = np.array(data)
    image = np.array(Image.fromarray((image).astype(np.uint8)).resize((224, 224)))
    image = image / 255.

    return image


def format_string(input_string):
    words = input_string.split('_')
    formatted_words = [word.capitalize() for word in words]
    formatted_string = ' '.join(formatted_words)
    return formatted_string


def main():
    st.title("GoldenBootHunter: Identifying golden boot winning players :soccer:")
    # # Display the football logo
    # logo_path = "logo.jpg"  # Adjust the path based on your actual file structure
    # st.image(logo_path, caption='SportLens', use_column_width=True)

    st.header("About")
    st.write("Introducing GoldenBootHunter: Your go-to app for identifying football's golden boot winners. Using a "
             "pretrained MobileNetV2 model and a dataset of 22 legends, it effortlessly recognizes iconic players "
             "from images. Explore the history of goal-scoring greatness with precise and reliable identification. "
             "Celebrate football excellence at your fingertips. Join the hunt for the golden boot stars!")
    st.subheader("Model Used: ")
    st.write("MobileNetV2(feature extractor,GlobalAveragePoolLayer,Dense Layer(256),DenseLayer(100),DenseLayer(22)")
    st.subheader("Dataset")
    st.markdown("[Golden Foot Football players Image Dataset](https://www.kaggle.com/datasets/balabaskar/golden-foot-football-players-image-dataset/data)")

    # Header with GitHub link
    st.header("GitHub Repository")
    st.markdown("[GoldenBootHunter Repository](https://github.com/Sadnan-Kawshik-015/GoldenBootHunter)")
    st.header("Player Image: ")
    # Upload image through Streamlit widget
    uploaded_image = st.file_uploader("Upload an image of a sport", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image.", use_column_width=True)
        pil_image = Image.open(uploaded_image)
        image = read_file_as_image(pil_image)
        img_batch = np.expand_dims(image, 0)
        predictions = MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])
        # print(type(predictions[0][0]))
        sorted_predictions = np.sort(predictions[0])

        # for element in sorted_predictions[-5:]:
        #  # Convert each element to percentage form
        #  percentage_value = element * 100.0
        #
        #  # Display the element in a Streamlit card-like format
        #  st.write(
        #   f"Original: {element:.4f} - Percentage: {percentage_value:.2f}%"
        #  )

        # Process the image (you can replace this with your own image processing logic)
        processed_text = predicted_class.title()

        # Display the processed text
        st.subheader("The Given Image is of: " + format_string(processed_text))
        st.subheader("Confidence: " + "{:.4f}".format(confidence * 100) + "%")
        # Footer with your name
    st.markdown("---")
    st.markdown("Created by: Sadnan Kibria Kawshik")
    st.markdown("[GitHub](https://github.com/Sadnan-Kawshik-015)")


def process_image(image):
    # Placeholder function for image processing
    # Replace this with your own image processing logic
    # For example, you can use OCR (Optical Character Recognition) to extract text from the image
    return "This is a placeholder for processed text."


if __name__ == "__main__":
    main()
