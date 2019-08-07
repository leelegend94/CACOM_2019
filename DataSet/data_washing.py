import os
import xarray as xr

TIME = 30

def washing(file_list,path):
	for file in file_list:
		if xr.open_dataset(path + '/' + file).to_array(dim='feature').transpose().to_pandas().shape[0] < TIME*60*4:
			#print(file)
			print(os.path.join(path,file))
			print(os.path.join(path,file+".invalid"))
			os.rename(os.path.join(path,file),os.path.join(path,file+".invalid"))


fg = os.listdir("./Fallgruppe_60")
cg = os.listdir("./Kontrollgruppe_60")

cg = list(filter(lambda x: x.endswith('nc'), cg))
fg = list(filter(lambda x: x.endswith('nc'), fg))
print("cont")
washing(cg,"./Kontrollgruppe_60")
print("fall")
washing(fg,"./Fallgruppe_60")