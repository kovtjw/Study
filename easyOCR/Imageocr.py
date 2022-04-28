import easyocr
import os
, os, matplotlib.pyplot as plt,cv2
from matplotlib import rcParams
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
from tqdm import tqdm
import numpy as np, pandas as pd, warnings, time
from selenium.webdriver.common.keys import Keys
from selenium import webdriver

# warnings.filterwarnings(action='ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

exit()

path = os.path.dirname(os.path.realpath(__file__))

rcParams['font.family'] = 'NanumGothic'

img_path = os.listdir(path + '/data')

img_list = []
results_list = []

for img in tqdm(img_path):
    
    draw_img = cv2.cvtColor(cv2.imread(f"{path}/data/{img}"),cv2.COLOR_BGR2RGB)
    # img = np.array(Image.open(f"{path}/data/{img}").rotate(-90))                        
    img_list.append(draw_img)
    
    reader = easyocr.Reader(['en','ko'],gpu=True)
    results = reader.readtext(f"{path}/data/{img}")
    results_list.append(results)

for img,results,img_name in zip(img_list,results_list,img_path):
    fig, ax = plt.subplots(1, figsize=(20, 20))                        
    
    for rects,text,acc in results:
        
        text = ax.text(rects[0][0],rects[0][1], text, color='black', fontsize=15, fontweight='bold')

        text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'), path_effects.Normal()])

        rect = patches.Polygon([rects[0], rects[1], rects[2], rects[3]], linewidth=5, edgecolor='red', facecolor='none')
        ax.add_patch(rect)    
    
    plt.imshow(img)                       
    plt.axis('off')
    plt.show()                       
    
    # plt.imsave(f'{path}/data/mt_{img_name}',img)
    # cv2.imwrite(f'{path}/data/cv_{img_name}',img)

options = webdriver.ChromeOptions()
options.add_experimental_option('excludeSwitches', ['enable-logging'])
driver = webdriver.Chrome('C://chromedriver.exe', options=options)



driver.get("https://translate.google.co.kr/?hl=ko")       
driver.maximize_window()

translate_list = []

for results in results_list:
    tr_list = [] 
    
    for rects,text,acc in results:
        elem = driver.find_element_by_xpath('/html/body/c-wiz/div/div[2]/c-wiz/div[2]/c-wiz/div[1]/div[2]/div[3]/c-wiz[1]/span/span/div/textarea') 
        elem.send_keys(f'{text}')
        time.sleep(3)
        try: 
            driver.find_element_by_xpath('/html/body/c-wiz/div/div[2]/c-wiz/div[2]/c-wiz/div[1]/div[2]/div[3]/c-wiz[1]/div[3]/div/div/div/span').click()
            time.sleep(3)
        except:
            pass
        tr = driver.find_element_by_xpath('/html/body/c-wiz/div/div[2]/c-wiz/div[2]/c-wiz/div[1]/div[2]/div[3]/c-wiz[2]/div[7]/div/div[1]').text
        tr_list.append(tr)
        driver.get("https://translate.google.co.kr/?hl=ko")
        time.sleep(2)
    
    translate_list.append(tr_list)
    driver.close()
        
for img,results,tr_list,img_path in zip(img_list,results_list,translate_list,img_path):
    fig, ax = plt.subplots(1, figsize=(20, 20))                        
    for r,tr_text in zip(results,tr_list):
        rects = r[0]
        text = tr_text
        text = ax.text(rects[0][0],rects[0][1], text, color='black', fontsize=15, fontweight='bold')

        text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'), path_effects.Normal()])

        rect = patches.Polygon([rects[0], rects[1], rects[2], rects[3]], linewidth=5, edgecolor='red', facecolor='none')
        ax.add_patch(rect)    
        
    plt.imshow(img)                       
    plt.axis('off')
    plt.show()   
    
    # plt.imsave(f'{path}/data/mt_{img_name}',img)
    # cv2.imwrite(f'{path}/data/cv_{img_name}',img)
    
# with open(path+'/data/test1.txt','a',encoding='utf-8') as f:
#     for results in results_list:
#         text = results[1]
#         f.write(f'{text}\n')