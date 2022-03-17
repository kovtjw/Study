




class custom_dataset(data.Dataset):
    	def __init__(self, img_path, gt_path, scale=0.25, length=512): 
     	# 생성자, 매개변수 이미지 위치, 가중치 위치, scale = feature map / image, 이미지 길이 512,
		super(custom_dataset, self).__init__() # 재호출
		self.img_files = [os.path.join(img_path, img_file) for img_file in sorted(os.listdir(img_path))] # listdir = 지정한 디렉토리 내의 모든 파일과 디렉토리의 리스를 리턴, 
		self.gt_files  = [os.path.join(gt_path, gt_file) for gt_file in sorted(os.listdir(gt_path))]   # sorted = 본체 리스트는 남겨두고, 정렬한 새로운 리스트를 반환
		self.scale = scale
		self.length = length

	def __len__(self):
		return len(self.img_files)

	def __getitem__(self, index):  # 하나
		with open(self.gt_files[index], 'r',encoding='UTF-8') as f: # 'r' = 파일을 읽기만 할 때 사용, 'ㄱ'
			lines = f.readlines()  # readlines 함수는 파일의 모든 줄을 읽어서 각각의 줄을 요소로 갖는 리스트로 돌려준다.
		vertices, labels = extract_vertices(lines)
		
		img = Image.open(self.img_files[index])
		img, vertices = adjust_height(img, vertices)  # 높이를 조절하다, adjust height of image to aug data 데이터 증강을 위해 높이를 조절함 + Image.BILINEAR(보간법) 사용
		img, vertices = rotate_img(img, vertices)  # 이미지를 회전시킨다. rotate image [-10, 10] degree to aug data 데이터 증강을 위해 회전시킴 + Image.BILINEAR(보간법) 사용
		img, vertices = crop_img(img, vertices, labels, self.length) # 이미지를 자르다, labels 가 1->텍스트가 있다, 0->없다, <numpy.ndarray, (n,)> 
		transform = transforms.Compose([transforms.ColorJitter(0.5, 0.5, 0.5, 0.25), \
                                        transforms.ToTensor(), \
                                        transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))])
										# ColorJitter = 밝기, 대비, 채도, 색조 등을 임의로 바꾸는 기능
										# ToTensor = datatype을 tensor로 변경
										# Normalize = 이미지는 0~255를 갖지만, tensor로 타입 변경 시 0~1사이로 바뀌게 된다 이를 nomalizer를 이용하여, -1~1사이 값으로 정규화 시킨다.
		score_map, geo_map, ignored_map = get_score_geo(img, vertices, labels, self.scale, self.length)
		return transform(img), score_map, geo_map, ignored_map
										# get_score_geo = 스코어 값, 위치 정보(좌표) 값을 생성
										# score_map = 텍스트가 있는 맵의 점수 > 0값이 아닌?
										# geo_map = 텍스트가 위치한 위치 정보
										# ignored_map = 텍스트가 없는 
	'''generate score gt and geometry gt
	Input:
		img     : PIL Image
		vertices: vertices of text regions <numpy.ndarray, (n,8)>
		labels  : 1->valid, 0->ignore, <numpy.ndarray, (n,)>
		scale   : feature map / image
		length  : image length
	Output:
		score gt, geo gt, ignored
	'''