class Convert_YCB:
    def __init__(self):
        self.conversion_dict, self.name_to_number_dict, self.number_to_name_dict = self.create_conversion_dict()

    def convert_name(self, input_string):
        # Return the converted string if available, otherwise return the original string
        return self.conversion_dict.get(input_string)

    def convert_number(self, input_value):
        # Determine if the input is a name or a number and convert accordingly
        if isinstance(input_value, str):
            return self.name_to_number_dict.get(input_value)
        elif isinstance(input_value, int):
            return self.number_to_name_dict.get(input_value)
        else:
            raise ValueError("The input should either be a string or an int")

    def get_object_list(self):
        return list(self.conversion_dict.keys())

    def get_desc_names_list(self):
        return list(self.conversion_dict.values())

    @staticmethod
    def create_conversion_dict():
        # Original mapping provided in the question
        mapping_text = """
        002_master_chef_can blue_cylindrical_can
        003_cracker_box red_cracker_cardbox
        004_sugar_box yellow_sugar_cardbox
        005_tomato_soup_can red_cylindrical_can
        006_mustard_bottle yellow_mustard_bottle
        007_tuna_fish_can round_fish_can
        008_pudding_box brown_jelly_cardbox
        009_gelatin_box red_jelly_cardbox
        010_potted_meat_can spam_rectangular_can
        011_banana banana
        019_pitcher_base blue_cup
        021_bleach_cleanser white_bleach_bottle
        024_bowl red_bowl
        025_mug red_cup
        035_power_drill drill
        036_wood_block wooden_block
        037_scissors scissors
        040_large_marker marker_pen
        051_large_clamp black_clamp
        052_extra_large_clamp big_black_clamp
        061_foam_brick red_rectangular_block
        """
        # Splitting the mapping text into lines and then into key-value pairs
        pairs = mapping_text.strip().split('\n')
        conversion_dict = {}
        name_to_number = {}
        number_to_name = {}

        for idx, pair in enumerate(pairs, start=1):
            key, value = pair.strip().split()
            conversion_dict[key] = value
            conversion_dict[value] = key  # This line allows bidirectional lookup
            name_to_number[key] = idx
            number_to_name[idx] = key

        return conversion_dict, name_to_number, number_to_name


if __name__ == '__main__':
    convert_string = Convert_YCB()
    assert convert_string.convert_name("002_master_chef_can") == "blue_cylindrical_can"
    assert convert_string.convert_name("blue_cylindrical_can") == "002_master_chef_can"
    assert convert_string.convert_name("banana") == "011_banana"
    assert convert_string.convert_name("011") is None
