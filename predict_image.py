import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image

MODEL_PATH = 'best_model.keras'
IMG_SIZE = (150, 150)

# Your full class list in the exact order used in training
class_names = [
    "abraham_grampa_simpson",
    "agnes_skinner",
    "apu_nahasapeemapetilon",
    "barney_gumble",
    "bart_simpson",
    "carl_carlson",
    "charles_montgomery_burns",
    "chief_wiggum",
    "cletus_spuckler",
    "comic_book_guy",
    "disco_stu",
    "edna_krabappel",
    "fat_tony",
    "gil",
    "groundskeeper_willie",
    "homer_simpson",
    "kent_brockman",
    "krusty_the_clown",
    "lenny_leonard",
    "lionel_hutz",
    "lisa_simpson",
    "maggie_simpson",
    "marge_simpson",
    "martin_prince",
    "mayor_quimby",
    "milhouse_van_houten",
    "miss_hoover",
    "moe_szyslak",
    "ned_flanders",
    "nelson_muntz",
    "otto_mann",
    "patty_bouvier",
    "principal_skinner",
    "professor_john_frink",
    "rainier_wolfcastle",
    "ralph_wiggum",
    "selma_bouvier",
    "sideshow_bob",
    "sideshow_mel",
    "snake_jailbird",
    "troy_mcclure",
    "waylon_smithers"
]

def load_model(model_path=MODEL_PATH):
    print(f"Loading model from {model_path}...")
    model = keras.models.load_model(model_path)
    return model

def predict_image(image_path, model, class_names):
    img = image.load_img(image_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction[0])
    predicted_class = class_names[predicted_index]

    print(f"Predicted class for '{image_path}': {predicted_class}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python predict_image.py <path_to_image>")
        return

    image_path = sys.argv[1]

    if not os.path.isfile(image_path):
        print(f"Error: The image path '{image_path}' does not exist or is not a file.")
        sys.exit(1)

    model = load_model()
    predict_image(image_path, model, class_names)

if __name__ == '__main__':
    main()