import easyocr
reader = easyocr.Reader(['en', 'kr'])
result = reader.readtext('D:\Study\easyOCR\data\3.jpg')