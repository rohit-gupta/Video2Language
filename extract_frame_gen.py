with open("vidnames.txt","r") as f:
	contents = f.readlines()

vidnames = [name.strip() for name in contents]

print "cd YouTube2Text/youtubeclips-dataset"

for name in vidnames:
	print "mkdir -p","frames/" + name.split(".")[0]
	print "ffmpeg -i", name, "-q:v 1", "frames/" + name.split(".")[0] + "/%05d.jpg -hide_banner"


print "cd ../.."
