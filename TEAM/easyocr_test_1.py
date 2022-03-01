import easyocr

reader = easyocr.Reader(['en','ko'])
results = reader.readtext('0CFB2A61F0E3AB6B6C27F89E07257201.jpg')

reader1 = easyocr.Reader(['en','ko'])
results1 = reader1.readtext('')
print(results)

