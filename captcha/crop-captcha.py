from PIL import Image

image = Image.open("images/cap")

area1=(26,0,48,37)
digit1 = image.crop(area1)
digit1.save("digits/1.png")

area2=(47,0,67,37)
digit2 = image.crop(area2)
digit2.save("digits/2.png")

area3=(67,0,87,37)
digit3 = image.crop(area3)
digit3.save("digits/3.png")

area4=(85,0,103,37)
digit4 = image.crop(area4)
digit4.save("digits/4.png")

area5=(102,0,123,37)
digit5 = image.crop(area5)
digit5.save("digits/5.png")



