import os

def delete_images_from_txt(txt_file):
  """Deletes images based on paths in the given text file.

  Args:
    txt_file: Path to the text file containing image paths.
  """

  with open(txt_file, 'r') as f:
    image_paths = f.readlines()

  for path in image_paths:
    # Remove leading/trailing whitespace
    path = path.strip()
    if os.path.exists(path):
      try:
        os.remove(path)
      except OSError as e:
        print(f"Error deleting {path}: {e}")

txt_file_path = 'C:\\Users\\Aravind M\\Desktop\\Everything\\College\\3rd Year\\6th sem\\Mini Project\\Project\\code\\ProblematicImages.txt'
delete_images_from_txt(txt_file_path)