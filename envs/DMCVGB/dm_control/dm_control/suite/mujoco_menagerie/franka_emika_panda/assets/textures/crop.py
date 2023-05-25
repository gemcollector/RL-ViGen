from PIL import Image
img = Image.open('city1.png')
region = img.crop((0,0,50,50))
region.save('city1_50.png')
# pix = img.getpixel((1, 1))
# print(pix)