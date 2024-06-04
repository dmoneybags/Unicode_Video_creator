import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import sys
import tensorflow as tf
import time
np.set_printoptions(threshold=sys.maxsize)

FONT_BOX_SIZE = 20
IMAGE_HEIGHT = 3672
IMAGE_WIDTH = 6500
USED_CHARCTERS = ([*"^*()_+=-=qwertyuiopasdfghjklzxcvbnm[]\;',./{}|:<>? "])
NUMBER_OF_USED_CHARACTERS = len(USED_CHARCTERS)
FONT_PATH = os.path.join(os.getcwd(), "ANDALEMO.ttf")
FONT = ImageFont.truetype(FONT_PATH, 24)
VIDEO_MATRIX_CHARACTERS = []

def upload_file():
    file_path = input("drag the jpeg file to the terminal: ")
    return file_path
def get_image_from_file_path(file_path):
    return Image.open(file_path)
def get_final_image_dimensions(image_original_width, image_original_height):
    if image_original_height > image_original_width:
        padding_height_needed = IMAGE_HEIGHT - image_original_height
        padding_width_needed = padding_height_needed * (image_original_height/image_original_width)
        final_width = int((image_original_width + padding_width_needed) * 1.83)
        print(final_width)
        assert(final_width <= IMAGE_WIDTH)
        return final_width, IMAGE_HEIGHT
    #unicode character is 1.83 times wider than it is tall
    padding_width_needed = IMAGE_WIDTH - image_original_width
    padding_height_needed = padding_width_needed * (image_original_height/image_original_width)
    #unicode character is 1.83 times wider than it is tall
    final_height = int((image_original_height + padding_height_needed)/1.83)
    assert(final_height <= IMAGE_HEIGHT)
    return IMAGE_WIDTH, int((image_original_height + padding_height_needed)/1.83)
def process_image(image, brightness_modifier = 1, contrast_modifier = 2, sharpness_modifier = 3):
    image = image.convert("L")
    #print(f"MEAN OF IMAGE BEFORE PROCESSING {np.mean(image)}")
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast_modifier)
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(sharpness_modifier)
    width, height = image.size
    #print(f"IMAGE HAS WIDTH OF {width} AND HEIGHT OF {height}")
    #unicode character is 1.83 times wider than it is tall
    final_width, final_height = get_final_image_dimensions(width, height)
    #print(f"IMAGE RESIZED TO {final_width} BY {final_height}")
    image = image.resize((final_width, final_height))
    image = tf.keras.utils.normalize(image, axis=0) * 255
    #subtract the mean to evenly space values between positive and negative
    #An average pixel value of -1.85 seems to work best
    image = image - (np.mean(image) * 1/brightness_modifier)
    #print(f"MEAN OF IMAGE AFTER PROCESSING {np.mean(image)}")
    return image
def get_numpy_array_for_letter(char):
    image = Image.new(mode = "L", size = (FONT_BOX_SIZE, FONT_BOX_SIZE))
    draw = ImageDraw.Draw(image)
    draw.text((3,-4), char, fill=255, font=FONT)
    arr = np.array(image)
    binary_arr = (arr > 0).astype(int)
    binary_arr[np.where(binary_arr == 0)] = -1
    if char == "_":
        binary_arr[-1] = np.array([1] * FONT_BOX_SIZE)
    return binary_arr
def get_character_matrix_filter():
    character_matrix_filter = np.zeros((FONT_BOX_SIZE, FONT_BOX_SIZE, 1, NUMBER_OF_USED_CHARACTERS))
    index = 0
    for char in USED_CHARCTERS:
        character_matrix_filter[:, :, :, index] = np.reshape(get_numpy_array_for_letter(char), (20, 20, 1))
        index += 1
    return character_matrix_filter
def get_convolution_scores(image_arr, filters):
    image_arr = np.reshape(image_arr, (1, np.shape(image_arr)[0], np.shape(image_arr)[1], 1)).astype(np.intc)
    filters = np.reshape(filters, (np.shape(filters)[0], np.shape(filters)[1], 1, NUMBER_OF_USED_CHARACTERS)).astype(np.intc)
    convolution = tf.nn.convolution(image_arr, filters, strides=20).numpy()
    strongest_filter_indices = np.argmax(convolution, axis=-1)
    return strongest_filter_indices
def gen_image(best_letters_indices):
    best_letters_indices = best_letters_indices.astype(int)
    text_picture = ""
    for row in best_letters_indices[0]:
        result_array = [USED_CHARCTERS[i] for i in row]
        text_picture = text_picture + ''.join(result_array) + "\n"
    return text_picture
def get_number_of_frames_from_video(file_path):
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count
def get_video_dimensions(file_path):
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    ret, frame = cap.read()
    cap.release()
    image = Image.fromarray(frame)
    width, height = image.size
    return width, height
def fill_video_matrix(file_path, filters, num_frames):
    print("FILLING VIDEO MATRIX")
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    frame_number = 0
    p = tf.keras.utils.Progbar(num_frames)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image = Image.fromarray(frame)
        image = process_image(image)
        best_letters_indices = get_convolution_scores(image, filters)
        VIDEO_MATRIX_CHARACTERS.append(gen_image(best_letters_indices))
        frame_number += 1
        p.add(1)
def current_milli_time():
    return round(time.time() * 1000)
def clear_stdout():
    print("\033c")
def play_video():
    #assuming 30 fps
    milliseconds_per_frame = 1000/30
    num_frames = len(VIDEO_MATRIX_CHARACTERS)
    timings = [milliseconds_per_frame * (i + 10) for i in range(num_frames)]
    start_time = current_milli_time()
    frame_num = 0
    clear_stdout()
    for frame in VIDEO_MATRIX_CHARACTERS:
        while current_milli_time() - start_time <= timings[frame_num]:
            time.sleep(0.015)
        clear_stdout()
        print(frame)
        frame_num +=1
if __name__ == "__main__":
    picture = False
    file = sys.argv[1]
    brightness_modifier = float(sys.argv[2])
    contrast_modifier = float(sys.argv[3])
    sharpness_modifier = float(sys.argv[4])
    filters = get_character_matrix_filter()
    file_path = os.getcwd() + "/" + file
    if picture:
        image = get_image_from_file_path(file_path)
        t1 = time.time()
        image = process_image(image, brightness_modifier=brightness_modifier, contrast_modifier=contrast_modifier, 
                            sharpness_modifier=sharpness_modifier)
        best_letters_indices = get_convolution_scores(image, filters)
        text_picture = gen_image(best_letters_indices)
        print(text_picture)
        print(f"Algorithm took {time.time() - t1} to run")
        quit()
    num_frames = get_number_of_frames_from_video(file_path)
    width, height = get_video_dimensions(file_path)
    final_width, final_height = get_final_image_dimensions(width, height)
    fill_video_matrix(file_path, filters, num_frames)
    play_video()


