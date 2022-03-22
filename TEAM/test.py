import easyocr

reader = easyocr.Reader(['en','ko'])
results = reader.readtext('11.jpg')

print(results)

