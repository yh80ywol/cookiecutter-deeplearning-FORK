import os
import tensorflow as tf

#config before start!
file_path = "../../data/raw/pictures/petimages_25000"
class_names = ["Cat", "Dog"]

#filter out badly-encoded images that do not feature the string "JFIF" in their header
num_skipped = 0
for folder_name in (class_names[0], class_names[1]):
    folder_path = os.path.join(file_path, folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
        finally:
            fobj.close()

        if not is_jfif:
            num_skipped += 1
            # Delete corrupted image
            os.remove(fpath)

print("Deleted %d images" % num_skipped)