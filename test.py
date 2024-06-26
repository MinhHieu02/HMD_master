def read_and_trim_file(file_path):
    # Step 1: Read the file content
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Step 2: Determine the midpoint
    midpoint = len(content) // 2

    # Step 3: Write only the first half back to the file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content[:midpoint])

    print(
        f"The file has been trimmed to half its original size. Original size: {len(content)} characters, new size: {midpoint} characters.")


# Example usage
file_path = '/home/ubuntu/Desktop/HMD_master/data/feature_data/race_train_large_features.json'
read_and_trim_file(file_path)
