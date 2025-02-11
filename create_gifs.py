from PIL import Image
from pathlib import Path
from tqdm import tqdm
import fnmatch
import os
import wandb


def paste_to_bg(x, bg):
	"adds image to background, but since .paste is in-place, we operate on a copy"
	bg2 = bg.copy()
	bg2.paste(x, x)  # paste returns None since it's in-place
	return bg2

def images_to_gif(image_fnames, fname):
	if not image_fnames: return
	image_fnames.sort(key=lambda x: int(x.name.split('_')[-2])) #sort by step

	frames = [Image.open(image) for image in image_fnames]
	max_x = 1024    # 1024x GIF is displayable within RAM limit of Colab free version. mp4s can go bigger.
	if frames[0].size[0] > max_x:
		print(f"Rescaling to ({max_x},..) so as not to exceed Colab RAM.")
		ratio = max_x/frames[0].size[0]
		newsize = [int(x*ratio) for x in frames[0].size]
		if newsize[1]%2 != 0: newsize[1] += 1 # wow ffmpeg hates odd dimensions!
		frames = [x.resize(newsize, resample=Image.BICUBIC) for x in frames]

	if frames[0].mode == 'RGBA':  # transparency goes black when saved as gif, so let's put it on white first
		bg = Image.new('RGBA', frames[0].size, (255, 255, 255))
		frames = [paste_to_bg(x,bg).convert('RGB').convert('P', palette=Image.ADAPTIVE) for x in frames]

	print("saving gif")
	frame_one = frames[0]
	frame_one.save(f'{fname}.gif', format="GIF", append_images=frames,
			   save_all=True, duration=DURATION, loop=0)

	print("making mp4")
	cmd = f"ffmpeg -loglevel error -i {f'{fname}.gif'} -vcodec libx264 -crf 25 -pix_fmt yuv420p {f'{fname}.mp4'}"
	os.system(cmd)
	if not os.path.exists(f'{fname}.mp4'):
		print(f"Failed to create mp4 file: {fname}.mp4\")\n")

def download_files(filenames_to_download, run):
	keys = set()
	print('Downloading Files')
	for file in tqdm(run.files()):
		if Path(file.name).is_file():
			continue
		if Path(file.name).name not in filenames_to_download:
			continue
		file.download()
	return keys

def make_and_display_gifs(run):
	extension = ".png"
	all_filenames = [Path(file.name).name for file in run.files() if file.name.endswith(extension)]
	keys = set([Path(fname).stem.split('_')[0] for fname in all_filenames])
	for key in keys:
		fnms = fnmatch.filter(all_filenames, f'{key}*{extension}')
		if not all([Path('./media/images/val/' + fn).is_file() for fn in fnms]):
			download_files(fnms, run)
		else:
			print('All files already downloaded')
	for key in keys:
		image_fnames = list(Path('./media/images/val/').glob(f'{key}*{extension}'))
		images_to_gif(image_fnames, key)

RUN_PATH = 'jack-morris/unsupervised_disc/75jh9hno' #@param {type:"string"}
DURATION = 200 #@param {type:"integer"}

api = wandb.Api()
run = api.run(RUN_PATH)
make_and_display_gifs(run)