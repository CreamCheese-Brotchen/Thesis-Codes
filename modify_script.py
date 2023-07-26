script_file = "~/projects/thesis-repo/sbatch_test.sh"

# Open the existing shell script in read mode
with open(script_file, 'r') as file:
    script_content = file.read()

# Replace the desired parameters with new values
new_tensorboard_comment = "test food101 vaeAug DEBUG"
# new_augmentation_type = "new_augmentation_type_value"
# new_simpleAugmentation_name = "new_simpleAugmentation_name_value"

# Modify the script content with the new parameter values
script_content = script_content.replace(
    '--tensorboard_comment "food101 vaeAug DEBUG"',
    f'--tensorboard_comment "{new_tensorboard_comment}"'
)

# script_content = script_content.replace(
#     '--augmentation_type vae',
#     f'--augmentation_type {new_augmentation_type}'
# )

# # Note: Modify this line according to your actual script content
# # This is just an example; you need to find the exact string to replace
# script_content = script_content.replace(
#     'your_existing_simpleAugmentation_name_value',
#     new_simpleAugmentation_name
# )

# Open the script in write mode and overwrite the content with the updated version
with open(script_file, 'w') as file:
    file.write(script_content)

print(f"Updated the content of '{script_file}'")