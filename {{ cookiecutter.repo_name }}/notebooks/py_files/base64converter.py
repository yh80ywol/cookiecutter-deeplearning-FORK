import base64, json

#Lesen der Base64.txt
with open('base64.json') as image_file:
    data=json.load(image_file)

#Convert .json to ascii
filename = 'some_image.jpeg'
data = data["file"]["data"].encode('ascii')

#Write jpeg/png ...
with open(filename, "wb") as f:
    f.write(base64.decodebytes(data))

