import easyocr 
reader = easyocr.Reader(['ko', 'en'])
result = reader.readtext("D:\Study\easyOCR\eval_detecting3.bmp") 

print(result)