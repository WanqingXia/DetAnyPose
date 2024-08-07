"""
convert.py

Author: Wanqing Xia
Email: wxia612@aucklanduni.ac.nz

This script contains the Convert_YCB class which is used for converting between different naming conventions
in the YCB dataset. The class provides methods for converting between the original object names and
their corresponding descriptions, as well as between object names and their numerical identifiers.

The class also provides methods for retrieving the list of all object names and their corresponding descriptions
in the dataset.

The conversion mappings are defined in a static method and stored in dictionaries for efficient lookup.
The class also includes some basic tests to verify the correctness of the conversions.
"""


class Convert_LM:
    def __init__(self):
        self.list = self.create_dict()

    def convert_number(self, input_value):
        # Determine if the input is a name or a number and convert accordingly
        if isinstance(input_value, str):
            return self.list.index(input_value)
        elif isinstance(input_value, int):
            return self.list[input_value]
        else:
            raise ValueError("The input should either be a string or an int")

    def get_object_list(self):
        return self.list

    @staticmethod
    def create_dict():
        # Original mapping provided in the question
        name_list = [
            "brown toy ape",
            "blue bench vise",
            "white ceramic bowl",
            "black camera",
            "plants watering kettle",
            "pink toy cat",
            "blue coffee cup",
            "green power drill",
            "yellow duck toy",
            "grey egg box",
            "white glue bottle",
            "blue hole puncher",
            "blue steam iron",
            "white lamp",
            "silver phone"
        ]

        return name_list


class Convert_LMO:
    def __init__(self):
        self.list = self.create_dict()

    def convert_number(self, input_value):
        # Determine if the input is a name or a number and convert accordingly
        if isinstance(input_value, str):
            return self.list.index(input_value)
        elif isinstance(input_value, int):
            return self.list[input_value]
        else:
            raise ValueError("The input should either be a string or an int")

    def get_object_list(self):
        return self.list

    @staticmethod
    def create_dict():
        # Original mapping provided in the question
        name_list = [
            "brown toy ape",
            "blue bench vise",
            "white ceramic bowl",
            "black camera",
            "plants watering kettle",
            "pink toy cat",
            "blue coffee cup",
            "green power drill",
            "yellow duck toy",
            "grey egg box",
            "white glue bottle",
            "blue hole puncher",
            "blue steam iron",
            "white lamp",
            "silver phone"
        ]

        return name_list


class Convert_HB:
    def __init__(self):
        self.list = self.create_dict()

    def convert_number(self, input_value):
        # Determine if the input is a name or a number and convert accordingly
        if isinstance(input_value, str):
            return self.list.index(input_value)
        elif isinstance(input_value, int):
            return self.list[input_value]
        else:
            raise ValueError("The input should either be a string or an int")

    def get_object_list(self):
        return self.list

    @staticmethod
    def create_dict():
        # Original mapping provided in the question
        name_list = [
            "brown toy bear",
            "blue bench vise",
            "blue toy car",
            "black toy cow",
            "white toy cow",
            "white coffee mug",
            "green power drill",
            "green toy bunny",
            "blue hole puncher",
            "brown ashtray",
            "brown round object",
            "black square block",
            "black handle",
            "white cylinder object",
            "black square object",
            "black valve",
            "black electrical component",
            "blue cake cardbox",
            "yellow minion toy",
            "colorful french bulldog",
            "silver cordless phone",
            "gray rhinoceros toy",
            "pug dog figurine",
            "vintage radio",
            "red toy car",
            "red toy motorcycle",
            "black high heel shoe",
            "stegosaurus toy",
            "yellow tea cardbox",
            "triceratops toy",
            "toy soldier figurine",
            "white toy car",
            "yellow toy bunny"
        ]

        return name_list


class Convert_HOPE:
    def __init__(self):
        self.list = self.create_dict()

    def convert_number(self, input_value):
        # Determine if the input is a name or a number and convert accordingly
        if isinstance(input_value, str):
            return self.list.index(input_value)
        elif isinstance(input_value, int):
            return self.list[input_value]
        else:
            raise ValueError("The input should either be a string or an int")

    def get_object_list(self):
        return self.list

    @staticmethod
    def create_dict():
        # Original mapping provided in the question
        name_list = [
            'blue soup can',
            'brown sauce bottle',
            'red butter block',
            'purple cheery can',
            'brown pudding box',
            'red white cookie box',
            'yellow corn can',
            'silver cheese block',
            'yellow snack box',
            'green beans can',
            'red ketchup bottle',
            'yellow macaroni box',
            'blue cap white bottle',
            'red milk box',
            'mushrooms can',
            'yellow mustard bottle'
            'yellow juice box',
            'yellow green can',
            'orange peach can',
            'green peas can',
            'yellow blue pineapple can',
            'blue popcorn box',
            'red raisins box',
            'green lid white bottle',
            'green spaghetti box',
            'red tomato can',
            'blue tuna can',
            'red yogurt bowl'
        ]

        return name_list


class Convert_YCBV:
    def __init__(self):
        self.list = self.create_dict()

    def convert_number(self, input_value):
        # Determine if the input is a name or a number and convert accordingly
        if isinstance(input_value, str):
            return self.list.index(input_value)
        elif isinstance(input_value, int):
            return self.list[input_value]
        else:
            raise ValueError("The input should either be a string or an int")

    def get_object_list(self):
        return self.list

    @staticmethod
    def create_dict():
        # Original mapping provided in the question
        name_list = [
            'blue cylindrical can',
            'red cracker cardbox',
            'yellow sugar cardbox',
            'red cylindrical can',
            'yellow mustard bottle',
            'tuna fish tin can',
            'brown jelly cardbox',
            'red jelly cardbox',
            'spam rectangular can',
            'banana',
            'blue cup',
            'white bleach bottle',
            'red bowl',
            'red cup',
            'drill',
            'wooden block',
            'scissors',
            'marker pen',
            'black clamp',
            'bigger black clamp',
            'red rectangular block'
        ]

        return name_list


if __name__ == '__main__':
    convert = Convert_YCBV()
    assert convert.convert_number("drill") == 14
    assert convert.convert_number(3) == "red cylindrical can"
