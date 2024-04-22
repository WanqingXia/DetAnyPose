import matplotlib.pyplot as plt

# Path to the file
file_path = './outputs/0000.txt'

# Initialize dictionaries
object_count = {}
mmdet_fail = {}
isolation_fail = {}
identification_fail = {}
success = {}

# Dictionary to hold references to the actual dictionaries based on the section names in the file
dict_references = {
    'object_count': object_count,
    'mmdet_sam_fail': mmdet_fail,
    'isolation_fail': isolation_fail,
    'identification_fail': identification_fail,
    'success': success
}

# Variable to keep track of which dictionary to populate
current_dict = None

# Open and read the file
with open(file_path, 'r') as file:
    for line in file:
        line = line.strip()
        if line in dict_references:  # This checks if the line is one of the dictionary names
            current_dict = dict_references[line]
        elif line and current_dict is not None:  # Check if line is not empty and we have set a current dictionary
            key, value = line.split(': ')
            current_dict[key] = int(value)  # Convert the value from string to integer

# Identify keys where the value in object_count is 0
keys_to_remove = [key for key, value in object_count.items() if value == 0]

# Remove these keys from all dictionaries
for key in keys_to_remove:
    del object_count[key]
    del mmdet_fail[key]
    del isolation_fail[key]
    del identification_fail[key]
    del success[key]

# Print the dictionaries to verify the contents
print("object_count:", object_count)
print("mmdet_fail:", mmdet_fail)
print("isolation_fail:", isolation_fail)
print("identification_fail:", identification_fail)
print("success:", success)
# List of keys to maintain consistent order across plots
keys = list(object_count.keys())

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

# Colors for different sections of the bar
colors = ['#77dd77', '#ff6961', '#fdfd96', '#84b6f4']
labels = ['success', 'mmdet_fail', 'isolation_fail', 'identification_fail']

# Adding the bars
for i, key in enumerate(keys):
    bottoms = 0
    total = object_count[key]
    success_rate = success[key] / total if total != 0 else 0

    # Adding each fail reason on top of each other and success at the bottom
    ax.bar(key, success[key], bottom=bottoms, color=colors[0], label='success' if i == 0 else "")
    bottoms += success[key]
    ax.bar(key, mmdet_fail[key], bottom=bottoms, color=colors[1], label='mmdet_fail' if i == 0 else "")
    bottoms += mmdet_fail[key]
    ax.bar(key, isolation_fail[key], bottom=bottoms, color=colors[2], label='isolation_fail' if i == 0 else "")
    bottoms += isolation_fail[key]
    ax.bar(key, identification_fail[key], bottom=bottoms, color=colors[3], label='identification_fail' if i == 0 else "")

    # Label at the top of each bar
    ax.text(i, total, f'Total: {total}\nRate: {success_rate:.0%}', ha='center', va='bottom')

# Labels and title
ax.set_ylabel('Count')
ax.set_title('Object Processing Results by Key')
ax.set_xticks(range(len(keys)))
ax.set_xticklabels(keys, rotation=90)
ax.legend()

# Show plot
plt.tight_layout()
plt.show()

