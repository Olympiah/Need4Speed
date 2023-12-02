import string
import easyocr

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}


def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'vehicle_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for frame_nmr in results.keys():
            for vehicle_id in results[frame_nmr].keys():
                print(results[frame_nmr][vehicle_id])
                if 'vehicle' in results[frame_nmr][vehicle_id].keys():
                    if 'license_plate' in results[frame_nmr][vehicle_id].keys():
                        if 'text' in results[frame_nmr][vehicle_id]['license_plate'].keys():
                            f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                                    vehicle_id,
                                                                    '[{} {} {} {}]'.format(
                                                                        results[frame_nmr][vehicle_id]['car']['bbox'][
                                                                            0],
                                                                        results[frame_nmr][vehicle_id]['car']['bbox'][
                                                                            1],
                                                                        results[frame_nmr][vehicle_id]['car']['bbox'][
                                                                            2],
                                                                        results[frame_nmr][vehicle_id]['car']['bbox'][
                                                                            3]),
                                                                    '[{} {} {} {}]'.format(
                                                                        results[frame_nmr][vehicle_id]['license_plate'][
                                                                            'bbox'][0],
                                                                        results[frame_nmr][vehicle_id]['license_plate'][
                                                                            'bbox'][1],
                                                                        results[frame_nmr][vehicle_id]['license_plate'][
                                                                            'bbox'][2],
                                                                        results[frame_nmr][vehicle_id]['license_plate'][
                                                                            'bbox'][3]),
                                                                    results[frame_nmr][vehicle_id]['license_plate'][
                                                                        'bbox_score'],
                                                                    results[frame_nmr][vehicle_id]['license_plate'][
                                                                        'text'],
                                                                    results[frame_nmr][vehicle_id]['license_plate'][
                                                                        'text_score'])
                                    )
        f.close()


def license_complies_format(text):
    """
    Check if the license plate text complies with the required format.
    The kenyan license plate has the first 3 as capital letters then 3 number and lastly a letter - cars

    Args:
        text (str): License plate text.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """
    if len(text) != 7:
        return False

    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
            (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
            (text[2] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
            (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
            (text[4] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
            (text[5] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
            (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
        return True
    else:
        return False


def format_license(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries
    in the beginning of the script

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """

    # We go through each char one by one and confirm that the format is okay, if not we replace
    # the char with the correct one e.g. the number 5 and S
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
               2: dict_char_to_int, 3: dict_char_to_int}
    for j in [0, 1, 2, 3, 4, 5, 6]:
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_


def read_license_plate(license_plate_crop):
    """
    Reads the license plate text from the given cropped image.

    Args:
        license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
    """

    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        # bbox-bounding box of plate, text - the text/characters, score - confidence of the detection i.e. license plate
        bbox, text, score = detection

        # converting the text to uppercase then eliminating whitespaces
        text = text.upper().replace(' ', '')

        # Here we check if the text complies with the standard format then format if necessary
        # and return the formatted text and confidence score
        if license_complies_format(text):
            return format_license(text), score

    return None, None


def get_vehicle(license_plate, vehicle_track_ids):
    """
    Retrieves the vehicle coordinates and track_ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates i.e. tracks

    Returns:
        tuple: Tuple containing the vehicle bbox/coordinates (x1, y1, x2, y2) and ID.
    """
    # unwrapping the license plate detections
    x1, y1, x2, y2, score, class_id = license_plate

    foundit = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, vehicle_id = vehicle_track_ids[j]
        # track = vehicle_track_ids[j]  # Assuming vehicle_track_ids is a list of Track objects
        # xcar1, ycar1, xcar2, ycar2 = track.to_ltrb()  # Accessing bounding box coordinates
        # vehicle_id = track.track_id  # Accessing the track ID

        # xcar1 = int(xcar1)
        # ycar1 = int(ycar1)
        # xcar2 = int(xcar2)
        # ycar2 = int(ycar2)

        # Recall: xcar1 is vehicle coordinate and x1 is license plate coordinate of that specific vehicle.

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            vehicle_idx = j
            foundit = True
            break
    # if we are able to confirm that a license plate does belong to a particular vehicle
    # then return the tracks of the vehicle i.e. bbox and track id else return -1(Null)
    if foundit:
        return vehicle_track_ids[vehicle_idx]
    else:
        return -1, -1, -1, -1, -1
